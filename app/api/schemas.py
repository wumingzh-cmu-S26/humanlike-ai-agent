"""Pydantic v2 request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------- Auth ----------
class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


# ---------- Chat ----------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Logical conversation id")
    message: str = Field(..., min_length=1, max_length=8192)
    personality: str | None = Field(default=None, description="Override personality profile")
    use_rag: bool = True
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    personality: str
    sentiment: dict[str, Any]
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int | None = None
    latency_ms: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


# ---------- Health ----------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    breakers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    services: dict[str, str] = Field(default_factory=dict)


# ---------- RAG ----------
class IngestRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(..., description="Each: {id, text, metadata}")


class IngestResponse(BaseModel):
    ingested: int
    skipped: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    top_k: int = Field(default=4, ge=1, le=50)


class SearchHit(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: int


# ---------- Voice ----------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    voice: str | None = None
    rate: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=0.0, ge=-12.0, le=12.0)


class DigitalHumanEvent(BaseModel):
    event_type: Literal["viseme", "emotion", "text", "audio_end"]
    timestamp_ms: int
    payload: dict[str, Any]
