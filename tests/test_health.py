"""Health endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():  # noqa: ANN201
    from app.main import create_app

    return TestClient(create_app())


def test_healthz(client):  # noqa: ANN001
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}


def test_readyz(client):  # noqa: ANN001
    r = client.get("/readyz")
    assert r.status_code == 200
