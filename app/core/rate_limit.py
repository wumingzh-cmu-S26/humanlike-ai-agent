"""Rate limiting using slowapi (token bucket per IP)."""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings


def _key_func(request) -> str:  # noqa: ANN001
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return f"jwt:{auth[7:][:32]}"
    return f"ip:{get_remote_address(request)}"


_settings = get_settings()
limiter = Limiter(
    key_func=_key_func,
    default_limits=[f"{_settings.rate_limit_per_minute}/minute"],
)
