"""Voice and digital-human endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.schemas import DigitalHumanEvent, TTSRequest
from app.core.security import get_current_user
from app.digital_human import build_dh_event_stream
from app.perception import get_sentiment_analyzer
from app.voice import get_tts

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/tts")
async def tts(body: TTSRequest, _user=Depends(get_current_user)) -> FileResponse:
    try:
        result = await get_tts().synthesize(body.text, body.voice, body.rate, body.pitch)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}") from e
    return FileResponse(
        path=result.audio_path,
        media_type="audio/mpeg",
        filename="speech.mp3",
        headers={"x-duration-ms": str(result.duration_ms)},
    )


@router.post("/digital-human", response_model=list[DigitalHumanEvent])
async def digital_human(
    body: TTSRequest, _user=Depends(get_current_user)
) -> list[DigitalHumanEvent]:
    try:
        result = await get_tts().synthesize(body.text, body.voice, body.rate, body.pitch)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}") from e
    sentiment = get_sentiment_analyzer().analyze(body.text)
    return build_dh_event_stream(result, sentiment["label"], body.text)
