"""rag_search tool — queries the hybrid retriever."""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.rag import get_retriever


class RagSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language question to search the knowledge base.")
    top_k: int = Field(default=4, ge=1, le=10)


async def _run(query: str, top_k: int = 4) -> str:
    retriever = get_retriever()
    hits, _ = await retriever.retrieve(query, top_k=top_k * 2, use_reranker=True)
    if not hits:
        return "No relevant documents found."
    lines = []
    for i, h in enumerate(hits[:top_k], 1):
        snippet = h["text"][:300].replace("\n", " ")
        src = h["metadata"].get("source", h["id"])
        lines.append(f"[{i}] (source={src}) {snippet}")
    return "\n".join(lines)


def make_rag_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_run,
        name="rag_search",
        description=(
            "Search the user's knowledge base for relevant documents. "
            "Use for questions about content the user has ingested."
        ),
        args_schema=RagSearchInput,
    )
