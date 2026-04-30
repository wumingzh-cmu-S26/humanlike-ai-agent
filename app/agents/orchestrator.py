"""Agent orchestrator: LangChain agent with OpenAI function calling, personality, memory, sentiment-aware perception."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, AsyncIterator

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.agents.personality import Personality, get_personality_registry
from app.core.circuit_breaker import call_async as breaker_call_async, get_breaker
from app.core.config import get_settings
from app.core.logging import get_logger
from app.memory import get_memory_manager
from app.perception import get_sentiment_analyzer
from app.rag.embeddings import get_embedding_provider
from app.tools import build_tool_registry

log = get_logger(__name__)


class AgentOrchestrator:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self.personalities = get_personality_registry()
        self.memory = get_memory_manager()
        self.sentiment = get_sentiment_analyzer()
        self.embedder = get_embedding_provider()
        self.tools = build_tool_registry()
        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            streaming=True,
        )
        self._fast_llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_fast_model,
            temperature=0.3,
            max_tokens=512,
        )
        self._executor_cache: dict[str, AgentExecutor] = {}
        self._breaker = get_breaker("openai_chat")

    # ---------- public ----------
    async def respond(
        self,
        session_id: str,
        message: str,
        personality_name: str | None = None,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        personality = self._resolve_personality(personality_name)
        sentiment = self.sentiment.analyze(message)

        q_emb = None
        if use_rag:
            try:
                q_emb = await self.embedder.embed_one(message)
            except Exception as e:
                log.warning("embed_failed_for_memory_query", error=str(e))

        ctx = self.memory.build_context(session_id, query_embedding=q_emb)
        executor = self._get_executor(personality)

        chat_history = self._format_history(ctx["history"])
        system_extras = self._format_extras(ctx["summary"], ctx["long_term"], sentiment)
        opening = personality.opening_for_sentiment(sentiment["label"])
        prefixed_input = (opening + message) if opening else message

        async def _call() -> dict[str, Any]:
            return await executor.ainvoke(
                {
                    "input": prefixed_input,
                    "chat_history": chat_history,
                    "system_extras": system_extras,
                }
            )

        result: dict[str, Any] = {}
        try:
            result = await breaker_call_async(self._breaker, _call)
            reply = result.get("output", "").strip()
        except Exception as e:
            log.exception("agent_invoke_failed", error=str(e))
            reply = "I'm having trouble reaching the model right now. Please try again in a moment."

        sources = self._extract_sources(result)

        # Persist turn
        self.memory.record_turn(session_id, message, reply)
        if q_emb is not None:
            try:
                self.memory.long.write(
                    session_id,
                    f"User: {message}\nAssistant: {reply}",
                    q_emb,
                    metadata={"sentiment": sentiment["label"]},
                )
            except Exception as e:
                log.warning("long_term_write_failed", error=str(e))

        # Async summary (non-blocking)
        try:
            await self.memory.summary.maybe_summarize(
                session_id, ctx["history"], self._summarize_with_fast_llm
            )
        except Exception:
            pass

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "reply": reply,
            "personality": personality.name,
            "sentiment": sentiment,
            "sources": sources,
            "latency_ms": latency_ms,
        }

    async def stream(
        self,
        session_id: str,
        message: str,
        personality_name: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Token-level streaming for SSE."""
        personality = self._resolve_personality(personality_name)
        sentiment = self.sentiment.analyze(message)
        ctx = self.memory.build_context(session_id, None)

        chat_history = self._format_history(ctx["history"])
        system_extras = self._format_extras(ctx["summary"], ctx["long_term"], sentiment)
        opening = personality.opening_for_sentiment(sentiment["label"])
        full_input = (opening + message) if opening else message

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", personality.system_prompt + "\n\n{system_extras}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | self._llm
        accum: list[str] = []
        yield {"event": "meta", "personality": personality.name, "sentiment": sentiment}
        async for chunk in chain.astream(
            {
                "input": full_input,
                "chat_history": chat_history,
                "system_extras": system_extras,
            }
        ):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                accum.append(piece)
                yield {"event": "token", "content": piece}
        full_reply = "".join(accum).strip()
        self.memory.record_turn(session_id, message, full_reply)
        yield {"event": "done", "reply": full_reply}

    # ---------- helpers ----------
    def _resolve_personality(self, name: str | None) -> Personality:
        n = name or self._settings.default_personality
        return self.personalities.get(n)

    def _get_executor(self, personality: Personality) -> AgentExecutor:
        if personality.name in self._executor_cache:
            return self._executor_cache[personality.name]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", personality.system_prompt + "\n\n{system_extras}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(self._llm, self.tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=6,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        self._executor_cache[personality.name] = executor
        return executor

    def _format_history(self, history: list[dict[str, str]]) -> list:
        out = []
        for m in history:
            if m["role"] == "user":
                out.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                out.append(AIMessage(content=m["content"]))
            else:
                out.append(SystemMessage(content=m["content"]))
        return out

    def _format_extras(
        self,
        summary: str,
        long_term: list[dict[str, Any]],
        sentiment: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        if summary:
            parts.append(f"Conversation summary so far:\n{summary}")
        if long_term:
            mem_lines = "\n".join(f"- {h['text'][:200]}" for h in long_term[:3])
            parts.append(f"Relevant long-term memories:\n{mem_lines}")
        parts.append(f"Detected user sentiment: {sentiment['label']} (score={sentiment['score']:.2f}).")
        return "\n\n".join(parts)

    def _extract_sources(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for step in result.get("intermediate_steps", []) or []:
            try:
                action, observation = step
                if getattr(action, "tool", "") == "rag_search":
                    sources.append({"tool": "rag_search", "snippet": str(observation)[:400]})
            except Exception:
                continue
        return sources

    async def _summarize_with_fast_llm(self, prompt: str) -> str:
        msg = await self._fast_llm.ainvoke(prompt)
        return getattr(msg, "content", str(msg))


@lru_cache
def get_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator()
