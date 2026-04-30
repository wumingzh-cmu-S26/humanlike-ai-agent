"""JWT auth + login flow."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():  # noqa: ANN201
    from app.main import create_app

    return TestClient(create_app())


def test_login_success(client):  # noqa: ANN001
    r = client.post("/auth/token", json={"username": "demo", "password": "demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_failure(client):  # noqa: ANN001
    r = client.post("/auth/token", json={"username": "demo", "password": "wrong"})
    assert r.status_code == 401


def test_protected_requires_auth(client):  # noqa: ANN001
    r = client.post("/rag/search", json={"query": "hello"})
    assert r.status_code == 401


def test_token_round_trip():  # noqa: ANN201
    from app.core.security import create_access_token, decode_token

    tok = create_access_token("alice", extra={"role": "admin"})
    payload = decode_token(tok)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"
