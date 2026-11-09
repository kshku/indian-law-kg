from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from . import config


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedder:
    """FastEmbed wrapper that uses CUDA when available and falls back to CPU."""

    dimension = config.TEXT_VECTOR_DIM

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        max_length: int = config.EMBEDDING_MAX_LENGTH,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ):
        from fastembed import TextEmbedding

        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cpu"
        self.device_error: str | None = None

        try:
            import onnxruntime as ort

            cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            cuda_available = False
        self.cuda_available = cuda_available

        if cuda_available:
            try:
                self.model = TextEmbedding(model_name=model_name, cuda=True)
                session = getattr(getattr(self.model, "model", None), "model", None)
                providers = session.get_providers() if session is not None else []
                if "CUDAExecutionProvider" in providers:
                    self.device = "cuda"
                else:
                    self.device_error = "CUDA provider was not selected"
            except Exception as exc:
                self.device_error = str(exc)

        if self.device != "cuda":
            self.model = TextEmbedding(model_name=model_name, cuda=False)

        tokenizer = getattr(getattr(self.model, "model", None), "tokenizer", None)
        if tokenizer is not None and max_length > 0:
            tokenizer.enable_truncation(max_length=max_length)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [
            [float(value) for value in vector]
            for vector in self.model.embed(
                list(texts), batch_size=self.batch_size
            )
        ]


@dataclass(frozen=True)
class VectorPoint:
    provision_iri: str
    document_id: str
    section_id: str
    title: str
    text: str
    document_title: str
    source_url: str | None
    text_vector: list[float]
    graph_vector: list[float]


@dataclass(frozen=True)
class ConceptPoint:
    concept_iri: str
    concept_id: str
    label: str
    definition: str
    aliases: tuple[str, ...]
    text_vector: list[float]


@dataclass(frozen=True)
class VectorHit:
    identifier: str
    score: float
    payload: dict


def point_id(identifier: str) -> str:
    return str(uuid5(NAMESPACE_URL, identifier))


class VectorStore:
    """Embedded Qdrant index for concepts, text, and graph vectors."""

    def __init__(
        self,
        path: Path = config.VECTOR_DB_PATH,
        text_dimension: int = config.TEXT_VECTOR_DIM,
        graph_dimension: int = config.GRAPH_VECTOR_DIM,
    ):
        self.path = Path(path)
        self.text_dimension = text_dimension
        self.graph_dimension = graph_dimension
        self.client = QdrantClient(path=str(self.path))

    def recreate(self) -> None:
        self._recreate_collection(
            config.PROVISION_COLLECTION,
            {
                "text": models.VectorParams(
                    size=self.text_dimension,
                    distance=models.Distance.COSINE,
                ),
                "graph": models.VectorParams(
                    size=self.graph_dimension,
                    distance=models.Distance.COSINE,
                ),
            },
        )
        self._recreate_collection(
            config.CONCEPT_COLLECTION,
            {
                "text": models.VectorParams(
                    size=self.text_dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )

    def _recreate_collection(self, name: str, vectors_config: dict) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
        )

    def upsert_provisions(self, points: Sequence[VectorPoint]) -> None:
        for start in range(0, len(points), 64):
            batch = points[start : start + 64]
            self.client.upsert(
                collection_name=config.PROVISION_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id(point.provision_iri),
                        vector={
                            "text": point.text_vector,
                            "graph": point.graph_vector,
                        },
                        payload={
                            "provision_iri": point.provision_iri,
                            "document_id": point.document_id,
                            "section_id": point.section_id,
                            "title": point.title,
                            "text": point.text,
                            "document_title": point.document_title,
                            "source_url": point.source_url,
                        },
                    )
                    for point in batch
                ],
            )

    def upsert_concepts(self, points: Sequence[ConceptPoint]) -> None:
        for start in range(0, len(points), 64):
            batch = points[start : start + 64]
            self.client.upsert(
                collection_name=config.CONCEPT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id(point.concept_iri),
                        vector={"text": point.text_vector},
                        payload={
                            "concept_iri": point.concept_iri,
                            "concept_id": point.concept_id,
                            "label": point.label,
                            "definition": point.definition,
                            "aliases": list(point.aliases),
                        },
                    )
                    for point in batch
                ],
            )

    def search_text(
        self,
        vector: Sequence[float],
        document_id: str | None = None,
        provision_iris: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[VectorHit]:
        if provision_iris is not None and not provision_iris:
            return []

        query_filter = self._filter(
            document_id=document_id,
            provision_iris=provision_iris,
        )
        result = self.client.query_points(
            collection_name=config.PROVISION_COLLECTION,
            query=list(vector),
            using="text",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            VectorHit(
                identifier=(point.payload or {}).get(
                    "provision_iri", str(point.id)
                ),
                score=float(point.score),
                payload=point.payload or {},
            )
            for point in result.points
        ]

    def search_concepts(
        self, vector: Sequence[float], limit: int = 3
    ) -> list[VectorHit]:
        result = self.client.query_points(
            collection_name=config.CONCEPT_COLLECTION,
            query=list(vector),
            using="text",
            limit=limit,
            with_payload=True,
        )
        return [
            VectorHit(
                identifier=(point.payload or {}).get(
                    "concept_iri", str(point.id)
                ),
                score=float(point.score),
                payload=point.payload or {},
            )
            for point in result.points
        ]

    def get_graph_vector(
        self, provision_iri: str
    ) -> list[float] | None:
        records = self.client.retrieve(
            collection_name=config.PROVISION_COLLECTION,
            ids=[point_id(provision_iri)],
            with_vectors=["graph"],
            with_payload=False,
        )
        if not records:
            return None
        vector = records[0].vector
        if isinstance(vector, dict):
            graph_vector = vector.get("graph")
        else:
            graph_vector = None
        if graph_vector is None:
            return None
        return [float(value) for value in graph_vector]

    def search_graph(
        self,
        vector: Sequence[float],
        limit: int = 10,
        exclude_provision_iri: str | None = None,
    ) -> list[VectorHit]:
        must_not = []
        if exclude_provision_iri is not None:
            must_not.append(
                models.HasIdCondition(
                    has_id=[point_id(exclude_provision_iri)]
                )
            )
        result = self.client.query_points(
            collection_name=config.PROVISION_COLLECTION,
            query=list(vector),
            using="graph",
            query_filter=models.Filter(must_not=must_not) if must_not else None,
            limit=limit,
            with_payload=True,
        )
        return [
            VectorHit(
                identifier=(point.payload or {}).get(
                    "provision_iri", str(point.id)
                ),
                score=float(point.score),
                payload=point.payload or {},
            )
            for point in result.points
        ]

    @staticmethod
    def _filter(
        document_id: str | None = None,
        provision_iris: Sequence[str] | None = None,
    ) -> models.Filter | None:
        must = []
        if document_id is not None:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )
        if provision_iris is not None:
            must.append(
                models.HasIdCondition(
                    has_id=[point_id(iri) for iri in provision_iris]
                )
            )
        return models.Filter(must=must) if must else None

    def close(self) -> None:
        self.client.close()
