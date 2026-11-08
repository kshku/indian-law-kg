"""Evaluate keyword-only and ontology-mediated retrieval."""

from indian_law_kg import config
from indian_law_kg.evaluation import evaluate, load_qrels
from indian_law_kg.kg import KnowledgeGraph
from indian_law_kg.retrieval import OntologyRetriever
from indian_law_kg.vector_store import FastEmbedder, VectorStore


class KeywordOnlyRetriever:
    """Baseline that uses only exact keyword search over the graph."""

    def __init__(self, kg):
        self.kg = kg

    def search(self, query, document_id=None, limit=10):
        results = []
        for result in self.kg.find_sections(query, document_id):
            row = dict(result)
            row["ontology_grounded"] = False
            row["match_type"] = "keyword_fallback"
            row["signals"] = ["keyword"]
            row["score"] = 0.0
            results.append(row)
            if len(results) >= limit:
                break
        return {"concepts": [], "results": results}


def main() -> None:
    if not config.VECTOR_DB_PATH.is_dir():
        raise SystemExit(
            f"Vector index not found: {config.VECTOR_DB_PATH}. "
            "Run `index` first."
        )
    if not config.GENERATED_ONTOLOGY_PATH.is_file():
        raise SystemExit(
            f"Generated ontology not found: {config.GENERATED_ONTOLOGY_PATH}. "
            "Run `build` first."
        )

    qrels = load_qrels(config.EVALUATION_QUERIES_PATH)
    kg = KnowledgeGraph(
        config.GENERATED_ONTOLOGY_PATH, config.ONTOLOGY_SCHEMA_PATH
    )
    store = VectorStore(config.VECTOR_DB_PATH)
    try:
        retriever = OntologyRetriever(kg, store, FastEmbedder())
        ontology = evaluate(retriever, qrels)
        keyword = evaluate(KeywordOnlyRetriever(kg), qrels)
    finally:
        store.close()

    print(f"Queries: {len(qrels)}")
    print()
    print(f"{'Metric':<26}{'keyword':>12}{'ontology':>12}")
    print("-" * 50)
    rows = [
        ("Concept top-1", "concept_top1", "score"),
        ("Concept top-3", "concept_top3", "score"),
        ("Recall@5", "recall@5", "score"),
        ("Recall@10", "recall@10", "score"),
        ("MRR", "mrr", "score"),
        ("nDCG@10", "ndcg@10", "score"),
        ("Citation accuracy", "citation_accuracy", "score"),
        ("Ontology-grounded", "ontology_grounded", "percent"),
        ("Median latency", "median_latency_ms", "ms"),
        ("p95 latency", "p95_latency_ms", "ms"),
    ]
    for label, key, kind in rows:
        if kind == "ms":
            left = f"{keyword[key]:.1f} ms"
            right = f"{ontology[key]:.1f} ms"
        elif kind == "percent":
            left = f"{keyword[key]:.0%}"
            right = f"{ontology[key]:.0%}"
        else:
            left = f"{keyword[key]:.3f}"
            right = f"{ontology[key]:.3f}"
        print(f"{label:<26}{left:>12}{right:>12}")


if __name__ == "__main__":
    main()
