"""Google Tasks tools."""
from __future__ import annotations

import asyncio
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
        raise RuntimeError(f"Google not connected for {username}.")
    return build("tasks", "v1", credentials=creds, cache_discovery=False)


class ListTasksInput(BaseModel):
    username: str = Field(...)


class AddTaskInput(BaseModel):
    username: str = Field(...)
    title: str = Field(..., min_length=1)
    notes: str = ""
    due_iso: str | None = Field(default=None, description="ISO-8601 datetime")


def _list_sync(username: str) -> list[dict[str, Any]]:
    svc = _service(username)
    lists = svc.tasklists().list(maxResults=1).execute().get("items", [])
    if not lists:
        return []
    list_id = lists[0]["id"]
    return svc.tasks().list(tasklist=list_id, showCompleted=False).execute().get("items", [])


async def _list(username: str) -> str:
    breaker = get_breaker("google_tasks")
    try:
        items = await breaker.call_async(asyncio.to_thread, _list_sync, username)
    except Exception as e:
        return f"Couldn't read tasks: {e}"
    if not items:
        return "No open tasks."
    return "\n".join(f"- {t.get('title')}" + (f" (due {t['due']})" if t.get("due") else "")
                     for t in items)


def _add_sync(username: str, title: str, notes: str, due_iso: str | None) -> dict[str, Any]:
    svc = _service(username)
    lists = svc.tasklists().list(maxResults=1).execute().get("items", [])
    if not lists:
        raise RuntimeError("No task list found.")
    list_id = lists[0]["id"]
    body: dict[str, Any] = {"title": title, "notes": notes}
    if due_iso:
        body["due"] = due_iso
    return svc.tasks().insert(tasklist=list_id, body=body).execute()


async def _add(username: str, title: str, notes: str = "", due_iso: str | None = None) -> str:
    breaker = get_breaker("google_tasks")
    try:
        t = await breaker.call_async(asyncio.to_thread, _add_sync, username, title, notes, due_iso)
    except Exception as e:
        return f"Couldn't add task: {e}"
    return f"Added task '{t.get('title')}' (id={t.get('id')})."


def make_list_tasks_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_list,
        name="tasks_list",
        description="List the user's open Google Tasks.",
        args_schema=ListTasksInput,
    )


def make_add_task_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_add,
        name="tasks_add",
        description="Create a Google Task. Use when the user asks to remember a to-do.",
        args_schema=AddTaskInput,
    )
