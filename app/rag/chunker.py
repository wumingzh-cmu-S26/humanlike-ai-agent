"""Recursive token-aware chunker for ingestion."""
from __future__ import annotations

import re

import tiktoken


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def _tokens(self, text: str) -> list[int]:
        if self._enc is None:
            return list(range(len(text) // 4))
        return self._enc.encode(text)

    def _decode(self, ids: list[int]) -> str:
        if self._enc is None:
            return ""
        return self._enc.decode(ids)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        if self._enc is None:
            return self._char_chunk(text)
        ids = self._tokens(text)
        out: list[str] = []
        i = 0
        while i < len(ids):
            window = ids[i : i + self.chunk_size]
            out.append(self._decode(window))
            i += self.chunk_size - self.chunk_overlap
        return out

    def _char_chunk(self, text: str) -> list[str]:
        sents = re.split(r"(?<=[.!?])\s+", text)
        out, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 > self.chunk_size * 4:
                out.append(cur.strip())
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            out.append(cur)
        return out
