"""Chroma metadata-aware vector retriever (filterable, persistent)."""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class ChromaRetriever:
    _COLLECTION = "rag_documents"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in ids],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, str, dict[str, Any]]]:
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
        except Exception as e:
            log.warning("chroma_query_failed", error=str(e))
            return []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out: list[tuple[str, float, str, dict[str, Any]]] = []
        for i, doc_id in enumerate(ids):
            score = 1.0 - (dists[i] if i < len(dists) else 0.0)
            out.append(
                (
                    doc_id,
                    float(score),
                    docs[i] if i < len(docs) else "",
                    metas[i] if i < len(metas) else {},
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()
