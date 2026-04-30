"""Verify breaker opens after configured failure count."""
from __future__ import annotations

import pytest
import pybreaker


def test_breaker_opens_after_failures():  # noqa: ANN201
    from app.core.circuit_breaker import get_breaker

    cb = get_breaker("test_unit_breaker")
    cb.fail_max = 3
    cb.reset_timeout = 60
    cb.close()

    @cb
    def fails():  # noqa: ANN201
        raise RuntimeError("nope")

    # pybreaker converts the failure that crosses the threshold into a
    # CircuitBreakerError, so accept either RuntimeError or CircuitBreakerError
    # for the first fail_max calls; subsequent calls must always be CircuitBreakerError.
    for _ in range(cb.fail_max):
        with pytest.raises((RuntimeError, pybreaker.CircuitBreakerError)):
            fails()

    with pytest.raises(pybreaker.CircuitBreakerError):
        fails()
    assert cb.current_state == "open"


def test_breaker_registry_singleton():  # noqa: ANN201
    from app.core.circuit_breaker import get_breaker

    a = get_breaker("singleton_test")
    b = get_breaker("singleton_test")
    assert a is b
