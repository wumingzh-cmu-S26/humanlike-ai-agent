"""Long-term episodic memory backed by Chroma — stores user-specific facts and salient turns."""
from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class LongTermMemory:
    _COLLECTION = "long_term_memory"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._top_k = settings.long_term_memory_top_k

    def write(
        self,
        session_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        doc_id = str(uuid.uuid4())
        meta = {"session_id": session_id, **(metadata or {})}
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[meta],
        )
        return doc_id

    def search(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self._top_k
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"session_id": session_id},
            )
        except Exception as e:
            log.warning("long_term_search_failed", error=str(e))
            return []
        out: list[dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "score": 1.0 - (dists[i] if i < len(dists) else 0.0),
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return out
