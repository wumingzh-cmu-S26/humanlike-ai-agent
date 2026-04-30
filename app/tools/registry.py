"""Build the tool list available to the agent."""
from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.google_calendar_tool import make_create_event_tool, make_list_events_tool
from app.tools.google_tasks_tool import make_add_task_tool, make_list_tasks_tool
from app.tools.messaging_tool import make_slack_send_tool, make_telegram_send_tool
from app.tools.rag_tool import make_rag_tool
from app.tools.time_tool import make_time_tool
from app.tools.web_search_tool import make_web_search_tool


def build_tool_registry() -> list[BaseTool]:
    return [
        make_rag_tool(),
        make_web_search_tool(),
        make_time_tool(),
        make_list_events_tool(),
        make_create_event_tool(),
        make_list_tasks_tool(),
        make_add_task_tool(),
        make_telegram_send_tool(),
        make_slack_send_tool(),
    ]
