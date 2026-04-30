"""web_search tool using DuckDuckGo (no API key required)."""
from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.circuit_breaker import get_breaker
from app.core.logging import get_logger

log = get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The web search query.")
    max_results: int = Field(default=5, ge=1, le=10)


async def _run(query: str, max_results: int = 5) -> str:
    breaker = get_breaker("web_search")

    def _sync_search() -> list[dict]:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        results = await breaker.call_async(asyncio.to_thread, _sync_search)
    except Exception as e:
        log.warning("web_search_failed", error=str(e))
        return f"Web search unavailable: {e}"

    if not results:
        return "No web results found."

    return "\n".join(
        f"[{i + 1}] {r.get('title', '')} — {r.get('href', '')}\n   {r.get('body', '')[:200]}"
        for i, r in enumerate(results)
    )


def make_web_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_run,
        name="web_search",
        description=(
            "Search the live web via DuckDuckGo. Use for current events, "
            "facts not in the knowledge base, or freshness-sensitive queries."
        ),
        args_schema=WebSearchInput,
    )
