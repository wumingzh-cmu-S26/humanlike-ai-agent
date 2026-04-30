"""Personality registry — loads YAML profiles into structured objects."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class Personality(BaseModel):
    name: str
    display_name: str
    description: str = ""
    voice: str = "en-US-JennyNeural"
    traits: dict[str, float] = Field(default_factory=dict)
    system_prompt: str
    greetings: list[str] = Field(default_factory=list)
    sentiment_responses: dict[str, str] = Field(default_factory=dict)

    def opening_for_sentiment(self, sentiment_label: str) -> str:
        return self.sentiment_responses.get(sentiment_label, "")


class PersonalityRegistry:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._cache: dict[str, Personality] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.directory.exists():
            log.warning("personality_dir_missing", path=str(self.directory))
            return
        for yml in self.directory.glob("*.yaml"):
            try:
                data = yaml.safe_load(yml.read_text(encoding="utf-8"))
                p = Personality.model_validate(data)
                self._cache[p.name] = p
            except Exception as e:
                log.warning("personality_load_failed", file=str(yml), error=str(e))

    def get(self, name: str) -> Personality:
        if name in self._cache:
            return self._cache[name]
        # fallback to default
        default = get_settings().default_personality
        if default in self._cache:
            return self._cache[default]
        raise KeyError(f"No personality '{name}' and no default '{default}' available.")

    def list(self) -> list[dict[str, Any]]:
        return [
            {"name": p.name, "display_name": p.display_name, "description": p.description}
            for p in self._cache.values()
        ]


@lru_cache
def get_personality_registry() -> PersonalityRegistry:
    return PersonalityRegistry(get_settings().personality_dir)
