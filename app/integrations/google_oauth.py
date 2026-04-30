"""Google OAuth 2.0 flow for Calendar + Tasks. Tokens persisted per user."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

try:
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
except ImportError:  # pragma: no cover
    Flow = None  # type: ignore
    Credentials = None  # type: ignore
    GoogleRequest = None  # type: ignore

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import get_current_user

log = get_logger(__name__)
router = APIRouter(prefix="/integrations/google", tags=["google"])

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]


def _client_config() -> dict[str, Any]:
    s = get_settings()
    return {
        "web": {
            "client_id": s.google_client_id,
            "client_secret": s.google_client_secret,
            "redirect_uris": [s.google_redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def _token_path(username: str) -> Path:
    s = get_settings()
    return Path(s.google_token_dir) / f"{username}.json"


def save_credentials(username: str, creds: Credentials) -> None:
    p = _token_path(username)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(creds.to_json(), encoding="utf-8")


def load_credentials(username: str) -> Credentials | None:
    if Credentials is None:
        return None
    p = _token_path(username)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    creds = Credentials.from_authorized_user_info(data, SCOPES)
    if creds.expired and creds.refresh_token and GoogleRequest is not None:
        try:
            creds.refresh(GoogleRequest())
            save_credentials(username, creds)
        except Exception as e:
            log.warning("google_token_refresh_failed", error=str(e))
            return None
    return creds


@router.get("/connect")
async def connect(_user=Depends(get_current_user)):  # noqa: ANN201
    if Flow is None:
        raise HTTPException(503, "google-auth-oauthlib not installed")
    s = get_settings()
    if not s.google_client_id:
        raise HTTPException(503, "Google OAuth not configured")
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = s.google_redirect_uri
    auth_url, _state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return {"auth_url": auth_url}


@router.get("/callback")
async def callback(request: Request):  # noqa: ANN201
    if Flow is None:
        raise HTTPException(503, "google-auth-oauthlib not installed")
    s = get_settings()
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = s.google_redirect_uri
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials

    # Identify the user by email from the id_token.
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    try:
        idinfo = google_id_token.verify_oauth2_token(
            creds.id_token, google_requests.Request(), s.google_client_id
        )
        email = idinfo.get("email", "unknown")
    except Exception:
        email = "unknown"

    save_credentials(email, creds)
    log.info("google_oauth_complete", email=email)
    return RedirectResponse(url="/healthz")
