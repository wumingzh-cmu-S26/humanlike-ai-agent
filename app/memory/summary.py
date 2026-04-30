"""Rolling summary memory — compresses older history into a running summary."""
from __future__ import annotations

from threading import Lock

import tiktoken

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class SummaryMemory:
    """Keeps a per-session running summary string. Compresses when token budget exceeded."""

    def __init__(self, token_limit: int = 2000) -> None:
        self.token_limit = token_limit
        self._summaries: dict[str, str] = {}
        self._lock = Lock()
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def get(self, session_id: str) -> str:
        with self._lock:
            return self._summaries.get(session_id, "")

    def set(self, session_id: str, summary: str) -> None:
        with self._lock:
            self._summaries[session_id] = summary

    def _count(self, text: str) -> int:
        if self._enc is None:
            return len(text) // 4
        return len(self._enc.encode(text))

    async def maybe_summarize(
        self,
        session_id: str,
        history: list[dict[str, str]],
        llm_summarize_fn,
    ) -> str:
        """If history is large, ask LLM to compress. Returns current summary."""
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        if self._count(joined) < self.token_limit:
            return self.get(session_id)
        prior = self.get(session_id)
        prompt = (
            "Update the running summary of this conversation. Be concise but keep "
            "user preferences, names, key facts, and unresolved threads.\n\n"
            f"Existing summary:\n{prior or '(none)'}\n\nRecent messages:\n{joined}\n\n"
            "Updated summary:"
        )
        try:
            new_summary = await llm_summarize_fn(prompt)
            self.set(session_id, new_summary.strip())
            log.info("summary_updated", session_id=session_id, chars=len(new_summary))
            return new_summary
        except Exception as e:
            log.warning("summary_failed", session_id=session_id, error=str(e))
            return prior
