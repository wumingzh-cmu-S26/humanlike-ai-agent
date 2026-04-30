"""Outbound Telegram and Slack send tools."""
from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.circuit_breaker import get_breaker
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


# ---------- Telegram ----------
class TelegramSendInput(BaseModel):
    chat_id: str = Field(..., description="Telegram chat id")
    text: str = Field(..., min_length=1, max_length=4096)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def _telegram_send(chat_id: str, text: str) -> str:
    s = get_settings()
    if not s.telegram_bot_token:
        return "Telegram not configured."
    breaker = get_breaker("telegram_out")

    async def _do() -> str:
        from telegram import Bot

        bot = Bot(token=s.telegram_bot_token)
        await bot.send_message(chat_id=chat_id, text=text)
        return "Sent."

    return await breaker.call_async(_do)


def make_telegram_send_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_telegram_send,
        name="telegram_send",
        description="Send a Telegram message to a known chat_id.",
        args_schema=TelegramSendInput,
    )


# ---------- Slack ----------
class SlackSendInput(BaseModel):
    channel: str = Field(..., description="Slack channel id or name (#general).")
    text: str = Field(..., min_length=1, max_length=4096)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def _slack_send(channel: str, text: str) -> str:
    s = get_settings()
    if not s.slack_bot_token:
        return "Slack not configured."
    breaker = get_breaker("slack_out")

    def _do() -> str:
        from slack_sdk import WebClient

        client = WebClient(token=s.slack_bot_token)
        resp = client.chat_postMessage(channel=channel, text=text)
        return f"ts={resp['ts']}"

    return await breaker.call_async(asyncio.to_thread, _do)


def make_slack_send_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_slack_send,
        name="slack_send",
        description="Send a Slack message to a channel.",
        args_schema=SlackSendInput,
    )
