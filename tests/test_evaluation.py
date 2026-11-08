import json

import pytest

from indian_law_kg.evaluation import (
    concept_accuracy_at_k,
    evaluate,
    load_qrels,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ranking_metric_helpers():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert recall_at_k(ranked, relevant, 0) == 0.0

    assert reciprocal_rank(ranked, relevant) == 0.5
    assert reciprocal_rank(["x", "y"], relevant) == 0.0

    assert ndcg_at_k(["a", "b", "c"], {"b"}, 3) == pytest.approx(
        1.0 / 1.584962500721156
    )
    assert ndcg_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(
        1.5 / (1.0 + 1.0 / 1.584962500721156)
    )
    assert ndcg_at_k(["a", "b"], {"b"}, 1) == 0.0

    assert concept_accuracy_at_k(["x", "y"], {"y"}, 1) == 0.0
    assert concept_accuracy_at_k(["x", "y"], {"y"}, 2) == 1.0
    assert concept_accuracy_at_k([], {"y"}, 3) == 0.0


def test_load_qrels_accepts_valid_entries(tmp_path):
    path = tmp_path / "qrels.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "definitions",
                    "document_id": "example-act-a",
                    "concepts": ["definitions"],
                    "relevant": ["2"],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_qrels(path) == [
        {
            "query": "definitions",
            "document_id": "example-act-a",
            "concepts": ["definitions"],
            "relevant": ["2"],
        }
    ]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({}, "no query"),
        ({"query": "x"}, "no document_id"),
        (
            {
                "query": "x",
                "document_id": "d",
                "concepts": [],
                "relevant": ["1"],
            },
            "invalid concepts",
        ),
        (
            {
                "query": "x",
                "document_id": "d",
                "concepts": ["c"],
                "relevant": [],
            },
            "invalid relevant",
        ),
    ],
)
def test_load_qrels_rejects_malformed_entries(tmp_path, entry, message):
    path = tmp_path / "qrels.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_qrels(path)


def test_load_qrels_rejects_non_list_file(tmp_path):
    path = tmp_path / "qrels.json"
    path.write_text(json.dumps({"query": "x"}), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list"):
        load_qrels(path)


class FakeRetriever:
    def __init__(self, kg):
        self.kg = kg
        self.calls = []

    def search(self, query, document_id=None, limit=10):
        self.calls.append((query, document_id, limit))
        return {
            "concepts": [{"concept_id": "definitions"}],
            "results": [
                {
                    "document_id": document_id,
                    "document_title": "Example Act A",
                    "source_url": None,
                    "section_id": "2",
                    "title": "Definitions",
                    "text": "Definitions text.",
                    "snippet": "Definitions Definitions text.",
                    "ontology_grounded": True,
                    "match_type": "concept",
                    "signals": ["concept"],
                    "score": 1.0,
                    "matched_concepts": ["definitions"],
                },
                {
                    "document_id": document_id,
                    "document_title": "Example Act A",
                    "source_url": None,
                    "section_id": "10",
                    "title": "Appeals",
                    "text": "Appeal text.",
                    "snippet": "Appeals Appeal text.",
                    "ontology_grounded": False,
                    "match_type": "keyword_fallback",
                    "signals": ["keyword"],
                    "score": 0.5,
                    "matched_concepts": [],
                },
            ],
        }


def test_evaluate_combines_metrics(kg):
    retriever = FakeRetriever(kg)
    qrels = [
        {
            "query": "definitions",
            "document_id": "example-act-a",
            "concepts": ["definitions"],
            "relevant": ["2"],
        }
    ]

    summary = evaluate(retriever, qrels, k_values=(1, 5))

    assert retriever.calls == [("definitions", "example-act-a", 5)]
    assert summary["queries"] == 1
    assert summary["concept_top1"] == 1.0
    assert summary["concept_top3"] == 1.0
    assert summary["recall@1"] == 1.0
    assert summary["recall@5"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["ndcg@10"] == 1.0
    assert summary["citation_accuracy"] == 1.0
    assert summary["ontology_grounded"] == 0.5
    assert summary["median_latency_ms"] >= 0.0
    assert summary["p95_latency_ms"] >= 0.0
    assert summary["per_query"][0]["ranked"] == ["2", "10"]
    assert summary["per_query"][0]["concepts"] == ["definitions"]
