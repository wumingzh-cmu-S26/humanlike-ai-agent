"""Source-of-truth document store. Persisted as JSONL for portability."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)


class DocumentStore:
    def __init__(self, path: str | Path = "./data/documents.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._docs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    self._docs[doc["id"]] = doc
                except json.JSONDecodeError:
                    log.warning("doc_store_skip_bad_line")

    def _persist(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for doc in self._docs.values():
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def upsert(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> bool:
        new = doc_id not in self._docs
        with self._lock:
            self._docs[doc_id] = {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
            }
            self._persist()
        return new

    def upsert_many(self, docs: list[dict[str, Any]]) -> tuple[int, int]:
        added = 0
        skipped = 0
        with self._lock:
            for d in docs:
                doc_id = d.get("id")
                text = d.get("text")
                if not doc_id or not text:
                    skipped += 1
                    continue
                is_new = doc_id not in self._docs
                self._docs[doc_id] = {
                    "id": doc_id,
                    "text": text,
                    "metadata": d.get("metadata", {}),
                }
                if is_new:
                    added += 1
            self._persist()
        return added, skipped

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._docs.get(doc_id)

    def all(self) -> list[dict[str, Any]]:
        return list(self._docs.values())

    def __len__(self) -> int:
        return len(self._docs)
