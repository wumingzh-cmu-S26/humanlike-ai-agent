"""Builds a digital-human event stream from a TTS result + sentiment.

Events are timestamped relative to audio start so a Unity/UE/web avatar can drive
mouth shapes (visemes) and emotional pose (emotion) in sync with audio playback.
"""
from __future__ import annotations

from app.api.schemas import DigitalHumanEvent
from app.voice.tts import TTSResult

_EMOTION_TIMELINE = {
    "negative": ["sad", "concerned", "neutral"],
    "positive": ["happy", "smile", "neutral"],
    "neutral": ["neutral"],
}


def build_dh_event_stream(
    tts: TTSResult,
    sentiment_label: str,
    text: str,
) -> list[DigitalHumanEvent]:
    events: list[DigitalHumanEvent] = []

    # Emotion key-frames spread across audio duration.
    poses = _EMOTION_TIMELINE.get(sentiment_label, ["neutral"])
    if poses:
        step = max(tts.duration_ms // len(poses), 1)
        for i, pose in enumerate(poses):
            events.append(
                DigitalHumanEvent(
                    event_type="emotion",
                    timestamp_ms=i * step,
                    payload={"pose": pose, "intensity": 0.7},
                )
            )

    # Word-by-word text events for captioning.
    for w in tts.word_boundaries:
        events.append(
            DigitalHumanEvent(
                event_type="text",
                timestamp_ms=int(w["audio_offset_ms"]),
                payload={"word": w["text"]},
            )
        )

    # Visemes for lip-sync.
    for v in tts.visemes:
        events.append(
            DigitalHumanEvent(
                event_type="viseme",
                timestamp_ms=int(v["audio_offset_ms"]),
                payload={"id": v["viseme_id"]},
            )
        )

    events.append(
        DigitalHumanEvent(
            event_type="audio_end",
            timestamp_ms=tts.duration_ms,
            payload={"text": text},
        )
    )

    events.sort(key=lambda e: e.timestamp_ms)
    return events
