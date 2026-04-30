"""Telegram webhook receiver — verifies secret token, routes to agent."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from app.agents import get_orchestrator
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(prefix="/integrations/telegram", tags=["telegram"])
log = get_logger(__name__)


@router.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, Any]:
    settings = get_settings()
    if settings.telegram_webhook_secret:
        if x_telegram_bot_api_secret_token != settings.telegram_webhook_secret:
            log.warning("telegram_webhook_bad_secret")
            raise HTTPException(401, "Bad secret")

    payload = await request.json()
    msg = payload.get("message") or payload.get("edited_message")
    if not msg:
        return {"ok": True, "ignored": True}

    text = msg.get("text") or ""
    chat_id = str(msg.get("chat", {}).get("id"))
    if not text or not chat_id:
        return {"ok": True, "ignored": True}

    log.info("telegram_inbound", chat_id=chat_id, len=len(text))
    orchestrator = get_orchestrator()
    result = await orchestrator.respond(
        session_id=f"telegram:{chat_id}",
        message=text,
    )

    # Reply via bot
    try:
        from telegram import Bot

        if settings.telegram_bot_token:
            bot = Bot(token=settings.telegram_bot_token)
            await bot.send_message(chat_id=chat_id, text=result["reply"])
    except Exception as e:
        log.warning("telegram_reply_failed", error=str(e))

    return {"ok": True}
