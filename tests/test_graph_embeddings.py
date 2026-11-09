import numpy as np
from rdflib import Graph

from indian_law_kg.graph_embeddings import (
    build_graph_embeddings,
    build_graph_matrix,
    embed_graph,
)
from indian_law_kg.kg import KnowledgeGraph


def cosine(left, right):
    left = np.array(left, dtype=float)
    right = np.array(right, dtype=float)
    return float(
        np.dot(left, right)
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def test_graph_matrix_is_square_symmetric_with_positive_self_loops(kg):
    provision_iris, matrix = build_graph_matrix(kg)

    assert len(provision_iris) == 8
    assert matrix.shape == (8, 8)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) > 0)


def test_graph_embedding_is_deterministic_and_canonical(kg):
    _, matrix = build_graph_matrix(kg)

    first = embed_graph(matrix, 4)
    second = embed_graph(matrix, 4)

    assert first.shape == (8, 4)
    assert np.allclose(first, second)
    for row in first:
        if np.any(row):
            assert row[int(np.argmax(np.abs(row)))] >= 0


def test_embedding_keeps_many_components_non_zero():
    embedding = embed_graph(np.eye(12), 4)

    assert np.all(np.linalg.norm(embedding, axis=1) > 0)


def test_related_provisions_are_more_similar_than_unrelated_ones(tmp_path):
    from indian_law_kg import config
    from indian_law_kg.documents import Document
    from indian_law_kg.ontology_generator import generate_ontology

    document = Document(
        id="act-a",
        title="Act A",
        pdf_path=config.DATA_DIR / "unused.pdf",
        source_url=None,
        expected_sections=3,
    )
    sections = [
        {
            "section_id": "2",
            "title": "Two",
            "text": "See section 3.",
            "start_page": 1,
        },
        {
            "section_id": "3",
            "title": "Three",
            "text": "See section 2.",
            "start_page": 1,
        },
        {
            "section_id": "4",
            "title": "Four",
            "text": "No references.",
            "start_page": 1,
        },
    ]
    graph = generate_ontology([(document, sections)], concepts=[])
    path = tmp_path / "graph.rdf"
    graph.serialize(destination=path, format="xml")
    kg = KnowledgeGraph(path)

    vectors = build_graph_embeddings(kg, 4)

    section_2 = vectors[
        "https://example.org/legal-ontology#act-a_Section2"
    ]
    section_3 = vectors[
        "https://example.org/legal-ontology#act-a_Section3"
    ]
    section_4 = vectors[
        "https://example.org/legal-ontology#act-a_Section4"
    ]

    assert cosine(section_2, section_3) > cosine(section_2, section_4)
    assert build_graph_embeddings(kg, 4) == vectors


def test_empty_graph_produces_empty_embeddings(tmp_path):
    path = tmp_path / "empty.rdf"
    Graph().serialize(destination=path, format="xml")
    kg = KnowledgeGraph(path)

    provision_iris, matrix = build_graph_matrix(kg)

    assert provision_iris == []
    assert matrix.shape == (0, 0)
    assert build_graph_embeddings(kg) == {}
