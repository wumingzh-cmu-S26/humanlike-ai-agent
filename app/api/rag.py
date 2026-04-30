"""RAG router — ingest and search."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.core.security import get_current_user
from app.rag import get_retriever

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, _user=Depends(get_current_user)) -> IngestResponse:
    retriever = get_retriever()
    added, skipped = await retriever.ingest(body.documents)
    return IngestResponse(ingested=added, skipped=skipped)


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, _user=Depends(get_current_user)) -> SearchResponse:
    retriever = get_retriever()
    t0 = time.perf_counter()
    hits, _ = await retriever.retrieve(body.query, top_k=body.top_k)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return SearchResponse(
        query=body.query,
        hits=[
            SearchHit(id=h["id"], text=h["text"], score=h["score"], metadata=h["metadata"])
            for h in hits
        ],
        latency_ms=latency_ms,
    )
