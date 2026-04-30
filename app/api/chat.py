"""Chat router: synchronous and SSE-streaming endpoints."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.agents import get_orchestrator
from app.api.schemas import ChatRequest, ChatResponse
from app.core.security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user=Depends(get_current_user),
) -> ChatResponse:
    orchestrator = get_orchestrator()
    result = await orchestrator.respond(
        session_id=body.session_id,
        message=body.message,
        personality_name=body.personality,
        use_rag=body.use_rag,
    )
    return ChatResponse(
        session_id=body.session_id,
        reply=result["reply"],
        personality=result["personality"],
        sentiment=result["sentiment"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    user=Depends(get_current_user),
) -> EventSourceResponse:
    orchestrator = get_orchestrator()

    async def gen() -> AsyncIterator[dict]:
        async for ev in orchestrator.stream(
            session_id=body.session_id,
            message=body.message,
            personality_name=body.personality,
        ):
            yield {"event": ev["event"], "data": json.dumps(ev)}

    return EventSourceResponse(gen())
