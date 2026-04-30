"""Unified memory manager: short-term + summary + long-term."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.summary import SummaryMemory

log = get_logger(__name__)


class MemoryManager:
    def __init__(self) -> None:
        settings = get_settings()
        self.short = ShortTermMemory(window_size=settings.memory_window_size)
        self.summary = SummaryMemory(token_limit=settings.memory_summary_token_limit)
        self._long_term: LongTermMemory | None = None  # lazy

    @property
    def long(self) -> LongTermMemory:
        if self._long_term is None:
            self._long_term = LongTermMemory()
        return self._long_term

    def record_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        self.short.add(session_id, "user", user_msg)
        self.short.add(session_id, "assistant", assistant_msg)

    def build_context(
        self,
        session_id: str,
        query_embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        history = self.short.get(session_id)
        summary = self.summary.get(session_id)
        long_hits: list[dict[str, Any]] = []
        if query_embedding is not None:
            try:
                long_hits = self.long.search(session_id, query_embedding)
            except Exception as e:
                log.warning("long_term_unavailable", error=str(e))
        return {"history": history, "summary": summary, "long_term": long_hits}


@lru_cache
def get_memory_manager() -> MemoryManager:
    return MemoryManager()
