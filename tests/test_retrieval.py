import pytest

from indian_law_kg.annotations import Concept
from indian_law_kg.retrieval import OntologyRetriever
from indian_law_kg.vector_store import VectorHit


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
            definition="A request for review of a decision.",
            aliases=("appeal", "appeals"),
        ),
        Concept(
            id="refund",
            label="refund",
            definition="Money returned to a buyer.",
            aliases=("refund",),
            broader=("consumer",),
        ),
    ]


class FakeStore:
    def __init__(self, concept_hits=None, text_hits=None):
        self.concept_hits = list(concept_hits or [])
        self.text_hits = list(text_hits or [])
        self.graph_vector = None
        self.similar_hits = []
        self.text_calls = []

    def search_concepts(self, vector, limit=3):
        return self.concept_hits

    def search_text(
        self,
        vector,
        document_id=None,
        provision_iris=None,
        limit=10,
    ):
        self.text_calls.append(
            {
                "document_id": document_id,
                "provision_iris": provision_iris,
            }
        )
        allowed = set(provision_iris or [])
        return [hit for hit in self.text_hits if hit.identifier in allowed]

    def get_graph_vector(self, provision_iri):
        return self.graph_vector

    def search_graph(self, vector, limit=10, exclude_provision_iri=None):
        return self.similar_hits


def hit(identifier, document_id, section_id, score=0.9):
    return VectorHit(
        identifier=identifier,
        score=score,
        payload={
            "document_id": document_id,
            "section_id": section_id,
        },
    )


def make_retriever(kg, fake_embedder, store=None):
    return OntologyRetriever(
        kg,
        store or FakeStore(),
        fake_embedder,
        concepts=concepts(),
    )


def test_resolve_concepts_prefers_aliases(kg, fake_embedder):
    retriever = make_retriever(kg, fake_embedder)

    resolved = retriever.resolve_concepts(
        "What do the definitions say about appeals?"
    )

    assert [
        (concept["concept_id"], concept["method"])
        for concept in resolved
    ] == [("definitions", "alias"), ("appeal", "alias")]


def test_resolve_concepts_uses_vectors_only_when_aliases_fail(
    kg, fake_embedder
):
    store = FakeStore(
        concept_hits=[
            VectorHit(
                identifier="https://example.org/legal-ontology#ConceptRefund",
                score=0.81,
                payload={"concept_id": "refund"},
            )
        ]
    )
    retriever = make_retriever(kg, fake_embedder, store)

    resolved = retriever.resolve_concepts("money returned")

    assert resolved == [
        {
            "concept_id": "refund",
            "label": "refund",
            "concept_iri": (
                "https://example.org/legal-ontology#ConceptRefund"
            ),
            "broader": ("consumer",),
            "method": "embedding",
            "score": 0.81,
        }
    ]

    store.concept_hits[0] = VectorHit(
        identifier="https://example.org/legal-ontology#ConceptRefund",
        score=0.1,
        payload={"concept_id": "refund"},
    )
    assert retriever.resolve_concepts("money returned") == []


def test_search_ranks_only_inside_ontology_candidates(kg, fake_embedder):
    store = FakeStore(
        text_hits=[
            hit(
                "https://example.org/legal-ontology#example-act-a_Section2A",
                "example-act-a",
                "2A",
            ),
            hit(
                "https://example.org/legal-ontology#example-act-b_Section99",
                "example-act-b",
                "99",
            ),
        ]
    )
    retriever = make_retriever(kg, fake_embedder, store)

    response = retriever.search("definitions", "example-act-a")

    assert [concept["concept_id"] for concept in response["concepts"]] == [
        "definitions"
    ]
    assert [
        (result["section_id"], result["match_type"])
        for result in response["results"]
    ] == [
        ("2", "concept"),
        ("2A", "related"),
        ("53-O", "related"),
    ]
    assert response["results"][0]["ontology_grounded"] is True
    assert response["results"][0]["signals"] == ["concept", "keyword"]
    assert response["results"][0]["matched_concepts"] == ["definitions"]
    assert response["results"][1]["signals"] == ["keyword", "vector"]
    assert all(
        result["document_id"] == "example-act-a"
        for result in response["results"]
    )
    assert store.text_calls[0]["provision_iris"] == [
        "https://example.org/legal-ontology#example-act-a_Section2",
        "https://example.org/legal-ontology#example-act-a_Section2A",
        "https://example.org/legal-ontology#example-act-a_Section53-O",
    ]


def test_search_without_resolved_concept_falls_back_to_keywords(
    kg, fake_embedder
):
    store = FakeStore()
    retriever = make_retriever(kg, fake_embedder, store)

    response = retriever.search("Neutral", "example-act-a")

    assert response["concepts"] == []
    assert response["results"]
    assert all(
        result["ontology_grounded"] is False
        for result in response["results"]
    )
    assert all(
        result["match_type"] == "keyword_fallback"
        for result in response["results"]
    )
    assert all(
        result["signals"] == ["keyword"]
        for result in response["results"]
    )
    assert store.text_calls == []


def test_search_returns_nothing_when_concept_has_no_annotated_sections(
    kg, fake_embedder
):
    store = FakeStore(
        concept_hits=[
            VectorHit(
                identifier="https://example.org/legal-ontology#ConceptRefund",
                score=0.9,
                payload={"concept_id": "refund"},
            )
        ],
        text_hits=[
            hit(
                "https://example.org/legal-ontology#example-act-b_Section99",
                "example-act-b",
                "99",
            )
        ],
    )
    retriever = make_retriever(kg, fake_embedder, store)

    response = retriever.search("money returned", "example-act-a")

    assert [concept["concept_id"] for concept in response["concepts"]] == [
        "refund"
    ]
    assert response["results"] == []
    assert store.text_calls == []


def test_search_respects_limit_and_empty_query(kg, fake_embedder):
    retriever = make_retriever(kg, fake_embedder)

    assert retriever.search("definitions", "example-act-a", limit=1)[
        "results"
    ][0]["section_id"] == "2"
    assert retriever.search("   ")["results"] == []
    assert retriever.search("definitions", limit=0)["results"] == []


def test_similar_returns_graph_neighbours(kg, fake_embedder):
    store = FakeStore()
    store.graph_vector = [1.0, 0.0, 0.0, 0.0]
    store.similar_hits = [
        hit(
            "https://example.org/legal-ontology#example-act-a_Section2A",
            "example-act-a",
            "2A",
            score=0.95,
        ),
        hit(
            "https://example.org/legal-ontology#example-act-b_Section99",
            "example-act-b",
            "99",
            score=0.5,
        ),
    ]
    retriever = make_retriever(kg, fake_embedder, store)

    results = retriever.similar("2", "example-act-a", limit=5)

    assert [
        (result["section_id"], result["match_type"])
        for result in results
    ] == [("2A", "graph_similar"), ("99", "graph_similar")]
    assert all(result["signals"] == ["graph"] for result in results)
    assert all(result["ontology_grounded"] for result in results)
    assert results[0]["score"] > results[1]["score"]


def test_similar_raises_for_missing_section_or_vector(kg, fake_embedder):
    store = FakeStore()
    retriever = make_retriever(kg, fake_embedder, store)

    with pytest.raises(ValueError, match="not found"):
        retriever.similar("missing", "example-act-a")
    with pytest.raises(ValueError, match="No graph vector"):
        retriever.similar("2", "example-act-a")
