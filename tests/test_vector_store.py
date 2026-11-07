from uuid import NAMESPACE_URL, uuid5

import pytest

from indian_law_kg.vector_store import (
    ConceptPoint,
    VectorPoint,
    VectorStore,
    point_id,
)


@pytest.fixture
def store(tmp_path):
    store = VectorStore(
        tmp_path / "qdrant",
        text_dimension=4,
        graph_dimension=3,
    )
    store.recreate()
    yield store
    store.close()


def provision(iri, document_id, section_id, text_vector, graph_vector):
    return VectorPoint(
        provision_iri=iri,
        document_id=document_id,
        section_id=section_id,
        title=f"Title {section_id}",
        text=f"Text {section_id}",
        document_title=f"{document_id} title",
        source_url=f"https://example.org/{document_id}",
        text_vector=text_vector,
        graph_vector=graph_vector,
    )


def add_provisions(store):
    store.upsert_provisions(
        [
            provision(
                "https://example.org/legal-ontology#act-a_Section2",
                "act-a",
                "2",
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ),
            provision(
                "https://example.org/legal-ontology#act-a_Section3",
                "act-a",
                "3",
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ),
            provision(
                "https://example.org/legal-ontology#act-b_Section2",
                "act-b",
                "2",
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
        ]
    )


def test_recreate_creates_both_collections(store):
    assert store.client.collection_exists("provisions")
    assert store.client.collection_exists("concepts")


def test_upsert_and_search_provisions_with_filters(store):
    add_provisions(store)

    hits = store.search_text([1.0, 0.0, 0.0, 0.0], limit=3)
    assert hits[0].identifier.endswith("act-a_Section2")
    assert hits[0].payload["document_id"] == "act-a"

    filtered = store.search_text(
        [1.0, 0.0, 0.0, 0.0], document_id="act-b", limit=3
    )
    assert [hit.payload["section_id"] for hit in filtered] == ["2"]

    restricted = store.search_text(
        [1.0, 0.0, 0.0, 0.0],
        provision_iris=[
            "https://example.org/legal-ontology#act-a_Section3"
        ],
        limit=3,
    )
    assert [hit.payload["section_id"] for hit in restricted] == ["3"]
    assert store.search_text([1.0, 0.0, 0.0, 0.0], provision_iris=[]) == []


def test_concept_vectors_and_payload(store):
    store.upsert_concepts(
        [
            ConceptPoint(
                concept_iri="https://example.org/legal-ontology#ConceptConsumer",
                concept_id="consumer",
                label="consumer",
                definition="A buyer.",
                aliases=("consumer", "consumers"),
                text_vector=[1.0, 0.0, 0.0, 0.0],
            ),
            ConceptPoint(
                concept_iri="https://example.org/legal-ontology#ConceptRefund",
                concept_id="refund",
                label="refund",
                definition="Money back.",
                aliases=("refund",),
                text_vector=[0.0, 1.0, 0.0, 0.0],
            ),
        ]
    )

    hits = store.search_concepts([1.0, 0.0, 0.0, 0.0], limit=1)

    assert hits[0].payload["concept_id"] == "consumer"
    assert hits[0].payload["aliases"] == ["consumer", "consumers"]


def test_graph_vector_retrieval_and_similarity(store):
    add_provisions(store)
    iri = "https://example.org/legal-ontology#act-a_Section2"

    assert store.get_graph_vector(iri) == [1.0, 0.0, 0.0]
    assert store.get_graph_vector("https://example.org/missing") is None

    hits = store.search_graph(
        [0.0, 1.0, 0.0], limit=3, exclude_provision_iri=iri
    )
    assert all(hit.identifier != iri for hit in hits)
    assert hits[0].identifier.endswith("act-a_Section3")


def test_point_ids_are_deterministic(store):
    iri = "https://example.org/legal-ontology#act-a_Section2"

    assert point_id(iri) == str(uuid5(NAMESPACE_URL, iri))
    assert point_id(iri) == point_id(iri)
