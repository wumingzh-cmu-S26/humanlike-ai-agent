"""Personality registry."""
from __future__ import annotations


def test_default_personalities_load():  # noqa: ANN201
    from app.agents.personality import get_personality_registry

    reg = get_personality_registry()
    names = {p["name"] for p in reg.list()}
    assert {"companion", "professional", "playful"}.issubset(names)


def test_personality_sentiment_opener():  # noqa: ANN201
    from app.agents.personality import get_personality_registry

    reg = get_personality_registry()
    p = reg.get("companion")
    assert "weighing" in p.opening_for_sentiment("negative") or p.opening_for_sentiment("negative")
    assert p.opening_for_sentiment("neutral") == ""
