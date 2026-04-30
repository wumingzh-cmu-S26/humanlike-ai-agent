"""Embedding provider — OpenAI text-embedding-3-small with circuit breaker."""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.circuit_breaker import get_breaker
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_DIM = 1536  # text-embedding-3-small


@lru_cache
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


class EmbeddingProvider:
    def __init__(self) -> None:
        self._breaker = get_breaker("openai_embeddings")
        self._model = get_settings().openai_embedding_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def _call() -> list[list[float]]:
            resp = await _client().embeddings.create(model=self._model, input=texts)
            return [d.embedding for d in resp.data]

        return await self._breaker.call_async(_call)

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed([text])
        return out[0] if out else [0.0] * _DIM


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()
