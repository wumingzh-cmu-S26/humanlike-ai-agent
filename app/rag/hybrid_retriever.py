"""Hybrid retrieval: BM25 + FAISS + Chroma fused with RRF, then cross-encoder rerank."""
from __future__ import annotations

import time
from functools import lru_cache
from threading import Lock
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.bm25_retriever import BM25Retriever
from app.rag.chroma_retriever import ChromaRetriever
from app.rag.document_store import DocumentStore
from app.rag.embeddings import get_embedding_provider
from app.rag.faiss_retriever import FaissRetriever
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import CrossEncoderReranker

log = get_logger(__name__)


class HybridRetriever:
    def __init__(self) -> None:
        settings = get_settings()
        self.store = DocumentStore()
        self.bm25 = BM25Retriever()
        self.faiss = FaissRetriever()
        self.chroma = ChromaRetriever()
        self.reranker = CrossEncoderReranker()
        self.embedder = get_embedding_provider()
        self.top_k = settings.rag_top_k
        self.rerank_top_k = settings.rag_rerank_top_k
        self.rrf_k = settings.rag_rrf_k
        self._faiss_path = settings.faiss_index_path
        self._lock = Lock()
        self._bootstrap()

    def _bootstrap(self) -> None:
        docs = self.store.all()
        if not docs:
            return
        self.bm25.index(docs)
        loaded = self.faiss.load(self._faiss_path, [d["id"] for d in docs])
        if loaded:
            log.info("faiss_loaded_from_disk", count=len(docs))
        else:
            log.info("faiss_will_rebuild_on_next_ingest", count=len(docs))

    async def ingest(self, documents: list[dict[str, Any]]) -> tuple[int, int]:
        added, skipped = self.store.upsert_many(documents)
        if not documents:
            return 0, skipped

        all_docs = self.store.all()
        ids = [d["id"] for d in all_docs]
        texts = [d["text"] for d in all_docs]
        metadatas = [d["metadata"] for d in all_docs]

        new_texts = [d["text"] for d in documents if d.get("id") and d.get("text")]
        new_ids = [d["id"] for d in documents if d.get("id") and d.get("text")]
        new_metas = [d.get("metadata", {}) for d in documents if d.get("id") and d.get("text")]
        new_embs = await self.embedder.embed(new_texts) if new_texts else []

        with self._lock:
            self.bm25.index(all_docs)
            if new_embs:
                self.chroma.upsert(new_ids, new_texts, new_embs, new_metas)
            all_embs_for_faiss = await self._collect_all_embeddings(ids, texts)
            self.faiss.build(ids, all_embs_for_faiss)
            try:
                self.faiss.save(self._faiss_path)
            except Exception as e:
                log.warning("faiss_save_failed", error=str(e))
        log.info("rag_ingest_complete", added=added, skipped=skipped, total=len(all_docs))
        return added, skipped

    async def _collect_all_embeddings(
        self, ids: list[str], texts: list[str]
    ) -> list[list[float]]:
        # Batch-embed everything for FAISS rebuild — small corpora friendly.
        batch_size = 64
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            out.extend(await self.embedder.embed(chunk))
        return out

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        timings: dict[str, float] = {}
        n = top_k or self.rag_initial_k()

        t0 = time.perf_counter()
        bm25_hits = self.bm25.search(query, top_k=n)
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        q_emb = await self.embedder.embed_one(query)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        faiss_hits = self.faiss.search(q_emb, top_k=n)
        timings["faiss_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        chroma_raw = self.chroma.search(q_emb, top_k=n)
        chroma_hits = [(r[0], r[1]) for r in chroma_raw]
        timings["chroma_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [bm25_hits, faiss_hits, chroma_hits],
            k=self.rrf_k,
        )
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        candidate_ids = [doc_id for doc_id, _ in fused[: self.top_k]]
        candidates: list[dict[str, Any]] = []
        for doc_id in candidate_ids:
            doc = self.store.get(doc_id)
            if doc:
                candidates.append({"id": doc_id, "text": doc["text"], "metadata": doc["metadata"]})

        if use_reranker and candidates:
            t0 = time.perf_counter()
            rer = self.reranker.rerank(
                query, [c["text"] for c in candidates], top_k=self.rerank_top_k
            )
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            results = [
                {
                    "id": candidates[i]["id"],
                    "text": candidates[i]["text"],
                    "metadata": candidates[i]["metadata"],
                    "score": score,
                }
                for i, score in rer
            ]
        else:
            results = [
                {**c, "score": float(self.top_k - rank)}
                for rank, c in enumerate(candidates[: self.rerank_top_k])
            ]

        return results, timings

    def rag_initial_k(self) -> int:
        return max(self.top_k * 2, 16)


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
