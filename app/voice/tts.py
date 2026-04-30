"""Azure Cognitive Services TTS with viseme/word-boundary capture for digital-human lipsync."""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.circuit_breaker import get_breaker
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class TTSResult:
    def __init__(
        self,
        audio_path: Path,
        visemes: list[dict[str, Any]],
        word_boundaries: list[dict[str, Any]],
        duration_ms: int,
    ) -> None:
        self.audio_path = audio_path
        self.visemes = visemes
        self.word_boundaries = word_boundaries
        self.duration_ms = duration_ms


class AzureTTS:
    def __init__(self) -> None:
        s = get_settings()
        self._key = s.azure_speech_key
        self._region = s.azure_speech_region
        self._default_voice = s.azure_speech_voice
        self._breaker = get_breaker("azure_tts")

    def _ssml(self, text: str, voice: str, rate: float, pitch: float) -> str:
        rate_pct = f"{int((rate - 1.0) * 100):+d}%"
        pitch_st = f"{pitch:+.1f}st"
        return f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="{voice}">
    <prosody rate="{rate_pct}" pitch="{pitch_st}">{text}</prosody>
  </voice>
</speak>"""

    def _synth_sync(
        self, text: str, voice: str, rate: float, pitch: float
    ) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], int]:
        if not self._key:
            raise RuntimeError("AZURE_SPEECH_KEY not set.")

        import azure.cognitiveservices.speech as speechsdk  # type: ignore

        cfg = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        out_path = Path(tempfile.gettempdir()) / f"tts-{uuid.uuid4().hex}.mp3"
        audio_cfg = speechsdk.audio.AudioOutputConfig(filename=str(out_path))
        synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)

        visemes: list[dict[str, Any]] = []
        word_boundaries: list[dict[str, Any]] = []

        def _on_viseme(evt) -> None:  # noqa: ANN001
            visemes.append(
                {"audio_offset_ms": evt.audio_offset // 10000, "viseme_id": evt.viseme_id}
            )

        def _on_word(evt) -> None:  # noqa: ANN001
            word_boundaries.append(
                {
                    "audio_offset_ms": evt.audio_offset // 10000,
                    "duration_ms": evt.duration.total_seconds() * 1000
                    if hasattr(evt.duration, "total_seconds")
                    else 0,
                    "text": evt.text,
                }
            )

        synth.viseme_received.connect(_on_viseme)
        synth.synthesis_word_boundary.connect(_on_word)

        ssml = self._ssml(text, voice, rate, pitch)
        result = synth.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")

        size = os.path.getsize(out_path)
        # Rough duration estimate from word boundaries
        duration_ms = (
            int(word_boundaries[-1]["audio_offset_ms"] + word_boundaries[-1]["duration_ms"])
            if word_boundaries
            else size // 4
        )
        return out_path, visemes, word_boundaries, duration_ms

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        rate: float = 1.0,
        pitch: float = 0.0,
    ) -> TTSResult:
        voice = voice or self._default_voice

        async def _do() -> TTSResult:
            audio_path, visemes, words, duration = await asyncio.to_thread(
                self._synth_sync, text, voice, rate, pitch
            )
            return TTSResult(audio_path, visemes, words, duration)

        return await self._breaker.call_async(_do)


@lru_cache
def get_tts() -> AzureTTS:
    return AzureTTS()
