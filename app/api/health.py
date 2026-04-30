"""Health & readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.schemas import HealthResponse
from app.core.circuit_breaker import all_breakers

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    breakers = all_breakers()
    degraded = any(b["state"] == "open" for b in breakers.values())
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        breakers=breakers,
        services={"api": "ok"},
    )


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, services={"api": "ready"})
