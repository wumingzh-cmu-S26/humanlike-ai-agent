"""Domain-level exceptions shared across the app."""
from __future__ import annotations


class AppError(Exception):
    """Base for all expected app errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class AuthError(AppError):
    status_code = 401
    code = "auth_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class UpstreamError(AppError):
    status_code = 502
    code = "upstream_error"


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"
