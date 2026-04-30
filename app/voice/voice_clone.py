"""Voice cloning hook — pluggable interface; Azure custom voice or other providers can implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)


class VoiceCloneProvider(ABC):
    @abstractmethod
    async def enroll(self, name: str, audio_samples: list[Path]) -> str:
        """Train/register a custom voice from samples. Returns a voice id usable in TTS."""

    @abstractmethod
    async def list_voices(self) -> list[dict]:
        """Return registered custom voices."""


class StubVoiceClone(VoiceCloneProvider):
    """Default stub that just registers a name -> samples mapping in memory.

    Replace with Azure Custom Voice / ElevenLabs / Coqui / etc. in production.
    """

    def __init__(self) -> None:
        self._registry: dict[str, list[Path]] = {}

    async def enroll(self, name: str, audio_samples: list[Path]) -> str:
        if not audio_samples:
            raise ValueError("Need at least one audio sample.")
        self._registry[name] = audio_samples
        log.info("voice_clone_enrolled", name=name, sample_count=len(audio_samples))
        return f"voice:{name}"

    async def list_voices(self) -> list[dict]:
        return [{"id": f"voice:{n}", "samples": len(s)} for n, s in self._registry.items()]


_default = StubVoiceClone()


def get_voice_clone() -> VoiceCloneProvider:
    return _default
