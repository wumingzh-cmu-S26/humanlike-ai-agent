"""Circuit breaker registry — wraps external calls (OpenAI, Google, Telegram, Slack, Azure)."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeVar

import pybreaker

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


async def call_async(
    breaker: pybreaker.CircuitBreaker,
    func: Callable[[], Awaitable[T]],
) -> T:
    """Run an async callable through the breaker's state machine.

    pybreaker.call_async requires tornado; this implementation uses pybreaker's
    sync state-tracking hooks while letting us await native coroutines.
    """
    if breaker.current_state == pybreaker.STATE_OPEN:
        raise pybreaker.CircuitBreakerError(f"Breaker {breaker.name} is open")
    try:
        result = await func()
    except Exception as exc:
        def _raise() -> None:
            raise exc
        try:
            breaker.call(_raise)
        except Exception:
            pass
        raise
    breaker.call(lambda: None)
    return result


class _BreakerListener(pybreaker.CircuitBreakerListener):
    def __init__(self, name: str) -> None:
        self.name = name

    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: Any,
        new_state: Any,
    ) -> None:
        log.warning(
            "circuit_breaker_state_change",
            breaker=self.name,
            old=getattr(old_state, "name", str(old_state)),
            new=getattr(new_state, "name", str(new_state)),
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: BaseException) -> None:
        log.warning("circuit_breaker_failure", breaker=self.name, error=str(exc))


_registry: dict[str, pybreaker.CircuitBreaker] = {}


def get_breaker(name: str) -> pybreaker.CircuitBreaker:
    if name in _registry:
        return _registry[name]
    settings = get_settings()
    cb = pybreaker.CircuitBreaker(
        fail_max=settings.circuit_breaker_fail_max,
        reset_timeout=settings.circuit_breaker_reset_timeout,
        listeners=[_BreakerListener(name)],
        name=name,
    )
    _registry[name] = cb
    return cb


def all_breakers() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "state": cb.current_state,
            "fail_counter": cb.fail_counter,
            "fail_max": cb.fail_max,
        }
        for name, cb in _registry.items()
    }
