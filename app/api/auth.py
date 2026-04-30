"""Auth router — issues JWTs for demo / dev users.

In production wire this to your real user store (DB, IAM, OIDC).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.api.schemas import LoginRequest, TokenResponse
from app.core.config import get_settings
from app.core.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

# Demo user store. Replace with persistent backend in production.
_DEMO_USERS: dict[str, str] = {"demo": hash_password("demo")}


def _issue(username: str, password: str) -> TokenResponse:
    hashed = _DEMO_USERS.get(username)
    if not hashed or not verify_password(password, hashed):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(username),
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/token", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    return _issue(body.username, body.password)


@router.post("/token/form", response_model=TokenResponse)
async def login_form(
    form: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    """OAuth2 password-flow form endpoint — used by the Swagger Authorize button."""
    return _issue(form.username, form.password)
