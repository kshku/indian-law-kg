import re
from collections.abc import Sequence

from . import config
from .annotations import Concept, concept_iri, load_concepts, match_concepts
from .kg import KnowledgeGraph, LEGAL
from .vector_store import Embedder, VectorStore

RRF_K = 60
CONCEPT_WEIGHT = 3.0
KEYWORD_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0
RELATED_WEIGHT = 0.25


def _snippet(title: str, text: str) -> str:
    return re.sub(r"\s+", " ", f"{title} {text[:200]}").strip()


def _key(result: dict) -> tuple[str, str]:
    return (result["document_id"], result["section_id"])


class OntologyRetriever:
    """Ontology-first retrieval with vectors used only inside grounded sets."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        store: VectorStore,
        embedder: Embedder,
        concepts: Sequence[Concept] | None = None,
    ):
        self.kg = kg
        self.store = store
        self.embedder = embedder
        self.concepts = list(
            concepts if concepts is not None else load_concepts()
        )
        self._concepts_by_id = {
            concept.id: concept for concept in self.concepts
        }

    def resolve_concepts(self, query: str) -> list[dict]:
        if not query.strip():
            return []

        matches = match_concepts(query, self.concepts)
        resolved = []
        seen = set()
        for match in matches:
            concept = match.concept
            if concept.id in seen:
                continue
            seen.add(concept.id)
            resolved.append(self._concept_result(concept, "alias", 1.0))
        if resolved:
            return resolved

        query_vector = self.embedder.embed([query])[0]
        hits = self.store.search_concepts(query_vector, limit=3)
        seen = set()
        for hit in hits:
            if hit.score < config.CONCEPT_SCORE_THRESHOLD:
                continue
            payload = hit.payload
            concept_id = payload.get("concept_id")
            if not concept_id or concept_id in seen:
                continue
            concept = self._concepts_by_id.get(concept_id)
            if concept is None:
                continue
            seen.add(concept_id)
            resolved.append(
                self._concept_result(concept, "embedding", hit.score)
            )
        return resolved

    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
    ) -> dict:
        if not query.strip():
            return {"concepts": [], "results": []}
        if limit <= 0:
            return {"concepts": [], "results": []}

        concepts = self.resolve_concepts(query)
        if not concepts:
            return {
                "concepts": [],
                "results": self._keyword_fallback(
                    query, document_id, limit
                ),
            }

        candidates = self.kg.find_sections_by_concepts(
            [concept["concept_id"] for concept in concepts],
            document_id,
        )
        if not candidates:
            return {"concepts": concepts, "results": []}

        direct = [
            candidate
            for candidate in candidates
            if candidate["match_type"] == "concept"
        ]
        related = [
            candidate
            for candidate in candidates
            if candidate["match_type"] == "related"
        ]
        candidate_by_key = {_key(candidate): candidate for candidate in candidates}
        direct_keys = {_key(candidate) for candidate in direct}
        candidate_iris = [
            candidate["provision_iri"]
            for candidate in candidates
            if candidate.get("provision_iri")
        ]

        keyword_results = self.kg.find_sections(query, document_id)
        keyword_rank = {}
        for result in keyword_results:
            key = _key(result)
            if key in candidate_by_key and key not in keyword_rank:
                keyword_rank[key] = len(keyword_rank) + 1

        query_vector = self.embedder.embed([query])[0]
        vector_hits = self.store.search_text(
            query_vector,
            document_id=document_id,
            provision_iris=candidate_iris,
            limit=max(1, len(candidate_iris)),
        )
        candidate_key_by_iri = {
            candidate["provision_iri"]: _key(candidate)
            for candidate in candidates
            if candidate.get("provision_iri")
        }
        vector_rank = {}
        for hit in vector_hits:
            key = candidate_key_by_iri.get(hit.identifier)
            if key is None:
                key = (
                    hit.payload.get("document_id"),
                    hit.payload.get("section_id"),
                )
            if key in candidate_by_key and key not in vector_rank:
                vector_rank[key] = len(vector_rank) + 1

        concept_rank = {
            _key(candidate): index
            for index, candidate in enumerate(direct, start=1)
        }
        related_rank = {
            _key(candidate): index
            for index, candidate in enumerate(related, start=1)
        }
        candidate_order = {
            _key(candidate): index
            for index, candidate in enumerate(candidates)
        }

        scored = []
        for key, candidate in candidate_by_key.items():
            score = 0.0
            signals = []
            if key in concept_rank:
                score += CONCEPT_WEIGHT / (
                    RRF_K + concept_rank[key]
                )
                signals.append("concept")
            if key in keyword_rank:
                score += KEYWORD_WEIGHT / (
                    RRF_K + keyword_rank[key]
                )
                signals.append("keyword")
            if key in vector_rank:
                score += VECTOR_WEIGHT / (
                    RRF_K + vector_rank[key]
                )
                signals.append("vector")
            if key in related_rank:
                score += RELATED_WEIGHT / (
                    RRF_K + related_rank[key]
                )
            scored.append(
                (score, key, candidate, signals)
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                0 if item[1] in direct_keys else 1,
                candidate_order[item[1]],
            )
        )
        keys = [key for _, key, _, _ in scored]
        hydrated = {
            _key(row): row for row in self.kg.get_sections(keys)
        }

        results = []
        for score, key, candidate, signals in scored:
            row = hydrated.get(key)
            if row is None:
                continue
            row = dict(row)
            row["snippet"] = _snippet(row["title"], row["text"])
            row["ontology_grounded"] = True
            row["match_type"] = (
                "concept" if key in direct_keys else "related"
            )
            row["related_via"] = candidate.get("related_via")
            row["signals"] = signals
            row["score"] = score
            row["matched_concepts"] = candidate.get("concepts", [])
            results.append(row)
            if len(results) >= limit:
                break

        return {"concepts": concepts, "results": results}

    def _keyword_fallback(
        self,
        query: str,
        document_id: str | None,
        limit: int,
    ) -> list[dict]:
        results = []
        for index, result in enumerate(
            self.kg.find_sections(query, document_id), start=1
        ):
            row = dict(result)
            row["ontology_grounded"] = False
            row["match_type"] = "keyword_fallback"
            row["signals"] = ["keyword"]
            row["score"] = KEYWORD_WEIGHT / (RRF_K + index)
            row["matched_concepts"] = []
            results.append(row)
            if len(results) >= limit:
                break
        return results

    def similar(
        self,
        section_id: str,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        if limit <= 0:
            return []
        resolved = self.kg._resolve_document_id(document_id)
        section = self.kg.get_section(section_id, resolved)
        if section is None:
            raise ValueError(
                f"Section {section_id} not found in document {resolved}"
            )

        seed_iri = str(
            LEGAL[f"{resolved}_Section{section_id}"]
        )
        vector = self.store.get_graph_vector(seed_iri)
        if vector is None:
            raise ValueError(
                f"No graph vector for section {section_id}. "
                "Run `index` first."
            )

        hits = self.store.search_graph(
            vector,
            limit=limit + 1,
            exclude_provision_iri=seed_iri,
        )
        hit_by_key = {}
        for hit in hits:
            key = (
                hit.payload.get("document_id"),
                hit.payload.get("section_id"),
            )
            if None not in key:
                hit_by_key.setdefault(key, hit)

        hydrated = {
            _key(row): row
            for row in self.kg.get_sections(list(hit_by_key))
        }
        results = []
        for key, hit in hit_by_key.items():
            row = hydrated.get(key)
            if row is None:
                continue
            row = dict(row)
            row["snippet"] = _snippet(row["title"], row["text"])
            row["ontology_grounded"] = True
            row["match_type"] = "graph_similar"
            row["related_via"] = None
            row["signals"] = ["graph"]
            row["score"] = hit.score
            row["matched_concepts"] = []
            results.append(row)
            if len(results) >= limit:
                break
        return results

    def _concept_result(
        self, concept: Concept, method: str, score: float
    ) -> dict:
        return {
            "concept_id": concept.id,
            "label": concept.label,
            "concept_iri": str(concept_iri(concept.id)),
            "broader": concept.broader,
            "method": method,
            "score": score,
        }
