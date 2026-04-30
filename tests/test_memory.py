"""Short-term memory & summary memory unit tests."""
from __future__ import annotations


def test_short_term_window():  # noqa: ANN201
    from app.memory.short_term import ShortTermMemory

    m = ShortTermMemory(window_size=2)  # window_size * 2 messages
    for i in range(10):
        m.add("s1", "user", f"msg {i}")
    history = m.get("s1")
    assert len(history) == 4  # window_size * 2
    assert history[-1]["content"] == "msg 9"


def test_short_term_isolated_per_session():  # noqa: ANN201
    from app.memory.short_term import ShortTermMemory

    m = ShortTermMemory()
    m.add("a", "user", "hello")
    m.add("b", "user", "world")
    assert m.get("a")[0]["content"] == "hello"
    assert m.get("b")[0]["content"] == "world"
