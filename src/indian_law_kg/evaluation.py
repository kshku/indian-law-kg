import json
import math
import statistics
import time
from pathlib import Path


def load_qrels(path: Path | str) -> list[dict]:
    """Load and validate hand-labelled retrieval queries."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("qrels must be a JSON list")

    qrels = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"qrel entry {index} must be an object")
        query = entry.get("query")
        document_id = entry.get("document_id")
        concepts = entry.get("concepts")
        relevant = entry.get("relevant")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"qrel entry {index} has no query")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"qrel entry {index} has no document_id")
        if (
            not isinstance(concepts, list)
            or not concepts
            or not all(
                isinstance(concept, str) and concept.strip()
                for concept in concepts
            )
        ):
            raise ValueError(f"qrel entry {index} has invalid concepts")
        if (
            not isinstance(relevant, list)
            or not relevant
            or not all(
                isinstance(section, str) and section.strip()
                for section in relevant
            )
        ):
            raise ValueError(f"qrel entry {index} has invalid relevant")
        qrels.append(
            {
                "query": query,
                "document_id": document_id,
                "concepts": concepts,
                "relevant": relevant,
            }
        )
    return qrels


def recall_at_k(
    ranked: list[str], relevant: set[str] | list[str], k: int
) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(ranked[:k]) & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(
    ranked: list[str], relevant: set[str] | list[str]
) -> float:
    relevant_set = set(relevant)
    for rank, identifier in enumerate(ranked, start=1):
        if identifier in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: list[str], relevant: set[str] | list[str], k: int
) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for rank, identifier in enumerate(ranked[:k], start=1):
        if identifier in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def concept_accuracy_at_k(
    ranked: list[str], relevant: set[str] | list[str], k: int
) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    return 1.0 if set(ranked[:k]) & relevant_set else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def evaluate(
    retriever, qrels: list[dict], k_values: tuple[int, ...] = (5, 10)
) -> dict:
    """Evaluate concept resolution and retrieval against hand labels."""
    per_query = []
    all_latencies = []
    total_results = 0
    grounded_results = 0
    citation_hits = 0

    for qrel in qrels:
        started = time.perf_counter()
        response = retriever.search(
            qrel["query"],
            qrel["document_id"],
            limit=max(k_values),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        all_latencies.append(latency_ms)

        concepts = response.get("concepts", [])
        concept_ids = [
            concept.get("concept_id")
            for concept in concepts
            if concept.get("concept_id")
        ]
        results = response.get("results", [])
        ranked = [result["section_id"] for result in results]

        keys = [
            (result["document_id"], result["section_id"])
            for result in results
            if result.get("document_id") and result.get("section_id")
        ]
        valid_keys = {
            (row["document_id"], row["section_id"])
            for row in retriever.kg.get_sections(keys)
        }
        citation_hits += sum(1 for key in keys if key in valid_keys)
        total_results += len(results)
        grounded_results += sum(
            1 for result in results if result.get("ontology_grounded")
        )

        metrics = {
            "query": qrel["query"],
            "document_id": qrel["document_id"],
            "concepts": concept_ids,
            "concept_top1": concept_accuracy_at_k(
                concept_ids, qrel["concepts"], 1
            ),
            "concept_top3": concept_accuracy_at_k(
                concept_ids, qrel["concepts"], 3
            ),
            "mrr": reciprocal_rank(ranked, qrel["relevant"]),
            "ndcg@10": ndcg_at_k(ranked, qrel["relevant"], 10),
            "latency_ms": latency_ms,
            "ranked": ranked,
        }
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(
                ranked, qrel["relevant"], k
            )
        per_query.append(metrics)

    summary = {
        "queries": len(qrels),
        "concept_top1": _mean(
            [query["concept_top1"] for query in per_query]
        ),
        "concept_top3": _mean(
            [query["concept_top3"] for query in per_query]
        ),
        "mrr": _mean([query["mrr"] for query in per_query]),
        "ndcg@10": _mean([query["ndcg@10"] for query in per_query]),
        "median_latency_ms": statistics.median(all_latencies)
        if all_latencies
        else 0.0,
        "p95_latency_ms": _percentile(all_latencies, 0.95),
        "citation_accuracy": (
            citation_hits / total_results if total_results else 0.0
        ),
        "ontology_grounded": (
            grounded_results / total_results if total_results else 0.0
        ),
        "per_query": per_query,
    }
    for k in k_values:
        summary[f"recall@{k}"] = _mean(
            [query[f"recall@{k}"] for query in per_query]
        )
    return summary
