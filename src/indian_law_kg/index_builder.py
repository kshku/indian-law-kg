from collections.abc import Sequence

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
) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed(texts[start : start + batch_size]))
    return vectors


def build_index(
    kg: KnowledgeGraph,
    embedder: Embedder,
    store: VectorStore,
    concepts: Sequence[Concept] | None = None,
    graph_dimension: int = config.GRAPH_VECTOR_DIM,
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

    concept_texts = [
        f"{concept.label}\n{' '.join(concept.aliases)}\n{concept.definition}"
        for concept in concepts
    ]
    concept_vectors = _embed_in_batches(embedder, concept_texts)
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

    sections = kg.list_sections_with_text()
    provision_iris = [
        str(LEGAL[f"{section['document_id']}_Section{section['section_id']}"])
        for section in sections
    ]
    graph_vectors = build_graph_embeddings(kg, graph_dimension)
    text_inputs = [
        f"{section['title']}\n{section['text']}" for section in sections
    ]
    text_vectors = _embed_in_batches(embedder, text_inputs)
    zero_graph = [0.0] * graph_dimension
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
