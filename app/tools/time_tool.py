"""get_current_time tool."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TimeInput(BaseModel):
    timezone: str = Field(
        default="UTC",
        description="IANA timezone, e.g. 'America/Los_Angeles', 'Asia/Shanghai', 'UTC'.",
    )


def _run(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
        timezone = "UTC"
    now = datetime.now(tz)
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {timezone} ({now.strftime('%A')})"


def make_time_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_run,
        name="get_current_time",
        description="Get the current date/time in any IANA timezone.",
        args_schema=TimeInput,
    )
