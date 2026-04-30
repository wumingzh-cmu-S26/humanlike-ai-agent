"""Shared pytest fixtures."""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-test-secret-test-secret")


@pytest.fixture
def temp_data_dir(monkeypatch):  # noqa: ANN001, ANN201
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("CHROMA_PERSIST_DIR", f"{tmp}/chroma")
        monkeypatch.setenv("FAISS_INDEX_PATH", f"{tmp}/faiss.index")
        from app.core.config import get_settings

        get_settings.cache_clear()  # type: ignore[attr-defined]
        yield tmp


@pytest.fixture
def auth_headers():  # noqa: ANN201
    from app.core.security import create_access_token

    token = create_access_token("demo")
    return {"Authorization": f"Bearer {token}"}
