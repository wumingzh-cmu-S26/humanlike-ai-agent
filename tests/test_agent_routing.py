"""Smoke-test that the tool registry includes the expected names."""
from __future__ import annotations


def test_tool_registry_names():  # noqa: ANN201
    from app.tools import build_tool_registry

    tools = build_tool_registry()
    names = {t.name for t in tools}
    assert {
        "rag_search",
        "web_search",
        "get_current_time",
        "calendar_list_events",
        "calendar_create_event",
        "tasks_list",
        "tasks_add",
        "telegram_send",
        "slack_send",
    }.issubset(names)


def test_time_tool_runs():  # noqa: ANN201
    from app.tools.time_tool import make_time_tool

    tool = make_time_tool()
    out = tool.invoke({"timezone": "UTC"})
    assert "UTC" in out
