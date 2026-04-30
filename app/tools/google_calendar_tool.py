"""Google Calendar tools."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.circuit_breaker import get_breaker
from app.core.logging import get_logger
from app.integrations.google_oauth import load_credentials

log = get_logger(__name__)


def _service(username: str):  # noqa: ANN201
    from googleapiclient.discovery import build

    creds = load_credentials(username)
    if creds is None:
        raise RuntimeError(
            f"Google not connected for {username}. Visit /integrations/google/connect."
        )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


class ListEventsInput(BaseModel):
    username: str = Field(..., description="The authenticated user's email/identifier.")
    days_ahead: int = Field(default=7, ge=1, le=60)


class CreateEventInput(BaseModel):
    username: str = Field(...)
    summary: str = Field(..., min_length=1)
    start_iso: str = Field(..., description="ISO-8601 start datetime, e.g. 2026-05-01T14:00:00Z")
    end_iso: str = Field(..., description="ISO-8601 end datetime")
    description: str = ""
    location: str = ""


def _list_events_sync(username: str, days_ahead: int) -> list[dict[str, Any]]:
    svc = _service(username)
    now = datetime.now(UTC)
    end = now + timedelta(days=days_ahead)
    res = (
        svc.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            maxResults=20,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return res.get("items", [])


async def _list_events(username: str, days_ahead: int = 7) -> str:
    import asyncio

    breaker = get_breaker("google_calendar")
    try:
        items = await breaker.call_async(asyncio.to_thread, _list_events_sync, username, days_ahead)
    except Exception as e:
        log.warning("calendar_list_failed", error=str(e))
        return f"Couldn't read calendar: {e}"
    if not items:
        return f"No events in the next {days_ahead} days."
    lines = []
    for ev in items:
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
        lines.append(f"- {start}: {ev.get('summary', '(no title)')}")
    return "\n".join(lines)


def _create_event_sync(
    username: str, summary: str, start_iso: str, end_iso: str, description: str, location: str
) -> dict[str, Any]:
    svc = _service(username)
    body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }
    return svc.events().insert(calendarId="primary", body=body).execute()


async def _create_event(
    username: str,
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
) -> str:
    import asyncio

    breaker = get_breaker("google_calendar")
    try:
        ev = await breaker.call_async(
            asyncio.to_thread,
            _create_event_sync,
            username,
            summary,
            start_iso,
            end_iso,
            description,
            location,
        )
    except Exception as e:
        log.warning("calendar_create_failed", error=str(e))
        return f"Couldn't create event: {e}"
    return f"Created event '{ev.get('summary')}' (id={ev.get('id')})."


def make_list_events_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_list_events,
        name="calendar_list_events",
        description="List the user's upcoming Google Calendar events.",
        args_schema=ListEventsInput,
    )


def make_create_event_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_create_event,
        name="calendar_create_event",
        description="Create a Google Calendar event. Confirm time/title with the user before invoking.",
        args_schema=CreateEventInput,
    )
