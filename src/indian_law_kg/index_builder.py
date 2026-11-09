from collections.abc import Callable, Sequence

from . import config
from .annotations import Concept, concept_iri, load_concepts
from .graph_embeddings import build_graph_embeddings
from .kg import LEGAL, KnowledgeGraph
from .vector_store import (
    ConceptPoint,
    Embedder,
    VectorPoint,
    VectorStore,
)


def _embed_in_batches(
    embedder: Embedder,
    texts: Sequence[str],
    batch_size: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed(texts[start : start + batch_size]))
        if on_progress is not None:
            on_progress(len(vectors), len(texts))
    return vectors


def build_index(
    kg: KnowledgeGraph,
    embedder: Embedder,
    store: VectorStore,
    concepts: Sequence[Concept] | None = None,
    graph_dimension: int = config.GRAPH_VECTOR_DIM,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Build concept, text, and graph vectors into the local store."""
    if embedder.dimension != store.text_dimension:
        raise ValueError(
            "Embedder dimension "
            f"{embedder.dimension} does not match store dimension "
            f"{store.text_dimension}"
        )

    if concepts is None:
        concepts = load_concepts()

    def report(stage: str, completed: int, total: int) -> None:
        if progress is not None:
            progress(stage, completed, total)

    concept_texts = [
        f"{concept.label}\n{' '.join(concept.aliases)}\n{concept.definition}"
        for concept in concepts
    ]
    concept_vectors = _embed_in_batches(
        embedder,
        concept_texts,
        on_progress=lambda completed, total: report(
            "concept embeddings", completed, total
        ),
    )
    report("writing concept vectors", 0, len(concepts))
    store.upsert_concepts(
        [
            ConceptPoint(
                concept_iri=str(concept_iri(concept.id)),
                concept_id=concept.id,
                label=concept.label,
                definition=concept.definition,
                aliases=concept.aliases,
                text_vector=vector,
            )
            for concept, vector in zip(concepts, concept_vectors)
        ]
    )
    report("writing concept vectors", len(concepts), len(concepts))

    sections = kg.list_sections_with_text()
    provision_iris = [
        str(LEGAL[f"{section['document_id']}_Section{section['section_id']}"])
        for section in sections
    ]
    report("graph vectors", 0, len(sections))
    graph_vectors = build_graph_embeddings(kg, graph_dimension)
    report("graph vectors", len(graph_vectors), len(sections))
    text_inputs = [
        f"{section['title']}\n{section['text']}" for section in sections
    ]
    text_vectors = _embed_in_batches(
        embedder,
        text_inputs,
        on_progress=lambda completed, total: report(
            "provision embeddings", completed, total
        ),
    )
    zero_graph = [0.0] * graph_dimension
    report("writing provision vectors", 0, len(sections))
    store.upsert_provisions(
        [
            VectorPoint(
                provision_iri=provision_iri,
                document_id=section["document_id"],
                section_id=section["section_id"],
                title=section["title"],
                text=section["text"],
                document_title=section["document_title"],
                source_url=section["source_url"],
                text_vector=text_vector,
                graph_vector=graph_vectors.get(provision_iri, zero_graph),
            )
            for section, provision_iri, text_vector in zip(
                sections, provision_iris, text_vectors
            )
        ]
    )
    report("writing provision vectors", len(sections), len(sections))

    return {
        "documents": len(
            {section["document_id"] for section in sections}
        ),
        "provisions": len(sections),
        "concepts": len(concepts),
        "text_vectors": len(text_vectors),
        "graph_vectors": len(graph_vectors),
        "concept_vectors": len(concept_vectors),
    }
