import pytest

from indian_law_kg.annotations import Concept
from indian_law_kg.index_builder import build_index
from indian_law_kg.vector_store import VectorStore


def concepts():
    return [
        Concept(
            id="definitions",
            label="definitions",
            definition="Statements that define terms.",
            aliases=("definitions",),
        ),
        Concept(
            id="appeal",
            label="appeal",
            definition="A request for review.",
            aliases=("appeal", "appeals"),
        ),
    ]


def test_build_index_writes_concept_text_and_graph_vectors(
    tmp_path, kg, fake_embedder
):
    store = VectorStore(
        tmp_path / "qdrant", text_dimension=4, graph_dimension=4
    )
    store.recreate()
    try:
        counts = build_index(
            kg,
            fake_embedder,
            store,
            concepts=concepts(),
            graph_dimension=4,
        )

        assert counts == {
            "documents": 2,
            "provisions": 8,
            "concepts": 2,
            "text_vectors": 8,
            "graph_vectors": 8,
            "concept_vectors": 2,
        }

        query = fake_embedder.embed(["Definitions\nDefinitions"])[0]
        hits = store.search_text(
            query, document_id="example-act-a", limit=3
        )
        assert hits
        assert all(
            hit.payload["document_id"] == "example-act-a"
            for hit in hits
        )

        concept_query = fake_embedder.embed(
            ["definitions\ndefinitions\nStatements that define terms."]
        )[0]
        concept_hits = store.search_concepts(concept_query, limit=1)
        assert concept_hits[0].payload["concept_id"] == "definitions"

        graph_vector = store.get_graph_vector(
            "https://example.org/legal-ontology#example-act-a_Section2"
        )
        assert graph_vector is not None
        assert len(graph_vector) == 4
    finally:
        store.close()


def test_build_index_rejects_dimension_mismatch(
    tmp_path, kg, fake_embedder
):
    store = VectorStore(
        tmp_path / "qdrant", text_dimension=4, graph_dimension=4
    )
    store.recreate()
    try:
        with pytest.raises(ValueError, match="Embedder dimension"):
            build_index(
                kg,
                type(fake_embedder)(3),
                store,
                concepts=concepts(),
                graph_dimension=4,
            )
    finally:
        store.close()
