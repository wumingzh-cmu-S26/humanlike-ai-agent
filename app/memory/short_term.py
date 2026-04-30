"""Sliding-window short-term memory keyed by session_id."""
from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any


class ShortTermMemory:
    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._store: dict[str, deque[dict[str, Any]]] = {}
        self._lock = Lock()

    def add(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            buf = self._store.setdefault(session_id, deque(maxlen=self.window_size * 2))
            buf.append({"role": role, "content": content})

    def get(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
