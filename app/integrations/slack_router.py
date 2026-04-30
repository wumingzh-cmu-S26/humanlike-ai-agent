"""Slack events webhook with signature verification + retry-safe handling."""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents import get_orchestrator
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(prefix="/integrations/slack", tags=["slack"])
log = get_logger(__name__)


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
    *,
    tolerance_seconds: int = 60 * 5,
) -> bool:
    if not signing_secret:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > tolerance_seconds:
        return False
    base = f"v0:{timestamp}:".encode() + body
    digest = hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(expected, signature)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def _post_reply(channel: str, text: str, thread_ts: str | None = None) -> None:
    import asyncio

    settings = get_settings()
    if not settings.slack_bot_token:
        return

    def _send() -> None:
        from slack_sdk import WebClient

        client = WebClient(token=settings.slack_bot_token)
        client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)

    await asyncio.to_thread(_send)


@router.post("/events")
async def slack_events(
    request: Request,
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
) -> dict[str, Any]:
    settings = get_settings()
    raw = await request.body()

    if not verify_slack_signature(
        raw,
        x_slack_request_timestamp or "",
        x_slack_signature or "",
        settings.slack_signing_secret,
    ):
        log.warning("slack_signature_invalid")
        raise HTTPException(401, "Invalid signature")

    payload = await request.json()
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event") or {}
    if event.get("type") not in {"app_mention", "message"}:
        return {"ok": True, "ignored": True}
    if event.get("bot_id"):
        return {"ok": True, "ignored": True}

    text = event.get("text") or ""
    channel = event.get("channel") or ""
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not text or not channel:
        return {"ok": True, "ignored": True}

    log.info("slack_inbound", channel=channel, len=len(text))
    orchestrator = get_orchestrator()
    result = await orchestrator.respond(
        session_id=f"slack:{channel}",
        message=text,
    )
    try:
        await _post_reply(channel, result["reply"], thread_ts=thread_ts)
    except Exception as e:
        log.warning("slack_post_reply_failed", error=str(e))
    return {"ok": True}
