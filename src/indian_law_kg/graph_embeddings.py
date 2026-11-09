import numpy as np
from rdflib import URIRef

from . import config
from .kg import QUERY_PREFIXES, LEGAL, KnowledgeGraph
from .sparql import run_select


def _provision_iri(document_id: str, section_id: str) -> URIRef:
    return LEGAL[f"{document_id}_Section{section_id}"]


def build_graph_matrix(
    kg: KnowledgeGraph,
) -> tuple[list[str], np.ndarray]:
    """Build a symmetric, normalised provision affinity matrix."""
    sections = kg.list_sections()
    provision_iris = [
        str(_provision_iri(section["document_id"], section["section_id"]))
        for section in sections
    ]
    if not provision_iris:
        return [], np.zeros((0, 0))

    index = {
        provision_iri: position
        for position, provision_iri in enumerate(provision_iris)
    }
    size = len(provision_iris)
    citation = np.zeros((size, size), dtype=float)

    citation_rows = run_select(
        kg.graph,
        QUERY_PREFIXES
        + "SELECT DISTINCT ?source ?target WHERE { "
        "?mention a legal:ReferenceMention ; "
        "legal:occursIn ?passage ; "
        "legal:referencesProvision ?target . "
        "?source legal:containsPassage ?passage . "
        "FILTER EXISTS { ?source a legal:Provision } "
        "FILTER EXISTS { ?target a legal:Provision } "
        "}",
    )
    for row in citation_rows:
        source = index.get(row.get("source"))
        target = index.get(row.get("target"))
        if source is None or target is None or source == target:
            continue
        citation[source, target] = 1.0
        citation[target, source] = 1.0

    concept_rows = run_select(
        kg.graph,
        QUERY_PREFIXES
        + "SELECT DISTINCT ?provision ?concept WHERE { "
        "?provision legal:containsPassage ?passage . "
        "?passage legal:mentionsConcept ?concept . "
        "FILTER EXISTS { ?provision a legal:Provision } "
        "FILTER EXISTS { ?concept a legal:LegalConcept } "
        "}",
    )
    concepts = sorted(
        {row["concept"] for row in concept_rows if row.get("concept")}
    )
    concept_index = {
        concept: position for position, concept in enumerate(concepts)
    }
    incidence = np.zeros((size, len(concepts)), dtype=float)
    for row in concept_rows:
        provision = index.get(row.get("provision"))
        concept = concept_index.get(row.get("concept"))
        if provision is not None and concept is not None:
            incidence[provision, concept] = 1.0

    if concepts:
        frequencies = incidence.sum(axis=0)
        weights = 1.0 / np.log2(2.0 + frequencies)
        weighted = incidence * weights
        concept_affinity = weighted @ weighted.T
    else:
        concept_affinity = np.zeros((size, size), dtype=float)

    affinity = citation + 0.5 * concept_affinity
    np.fill_diagonal(affinity, 0.0)
    # Self-loops keep every provision in the graph embedding and make
    # directly related provisions share neighbourhood features.
    affinity = affinity + np.eye(size)

    degrees = affinity.sum(axis=1)
    inverse_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    normalised = (
        inverse_sqrt[:, None] * affinity * inverse_sqrt[None, :]
    )
    return provision_iris, normalised


def embed_graph(matrix: np.ndarray, dimension: int) -> np.ndarray:
    """Embed local graph structure with a deterministic random projection.

    The one- and two-hop neighbourhoods are projected into ``dimension``
    dimensions. This preserves local similarity without letting the leading
    component of a large disconnected graph dominate every provision vector.
    """
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], dimension), dtype=float)

    features = matrix + 0.5 * (matrix @ matrix)
    projection = np.random.default_rng(42).standard_normal(
        (matrix.shape[1], dimension)
    )
    embedding = features @ projection / np.sqrt(max(dimension, 1))

    norms = np.linalg.norm(embedding, axis=1)
    positive = norms > 0
    embedding[positive] /= norms[positive, None]

    for row in range(embedding.shape[0]):
        values = embedding[row]
        if not np.any(values):
            continue
        pivot = int(np.argmax(np.abs(values)))
        if values[pivot] < 0:
            embedding[row] = -values
    return embedding


def build_graph_embeddings(
    kg: KnowledgeGraph,
    dimension: int = config.GRAPH_VECTOR_DIM,
) -> dict[str, list[float]]:
    """Build deterministic graph vectors keyed by provision IRI."""
    provision_iris, matrix = build_graph_matrix(kg)
    if not provision_iris:
        return {}
    embedding = embed_graph(matrix, dimension)
    return {
        provision_iri: [float(value) for value in embedding[position]]
        for position, provision_iri in enumerate(provision_iris)
    }
