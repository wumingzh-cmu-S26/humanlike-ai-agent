"""Slack request signature verification."""
from __future__ import annotations

import hashlib
import hmac
import time


def _sign(body: bytes, ts: str, secret: str) -> str:
    base = f"v0:{ts}:".encode() + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def test_valid_signature():  # noqa: ANN201
    from app.integrations.slack_router import verify_slack_signature

    body = b'{"event":"hi"}'
    ts = str(int(time.time()))
    secret = "shh"
    sig = _sign(body, ts, secret)
    assert verify_slack_signature(body, ts, sig, secret)


def test_tampered_body_rejected():  # noqa: ANN201
    from app.integrations.slack_router import verify_slack_signature

    body = b'{"event":"hi"}'
    ts = str(int(time.time()))
    secret = "shh"
    sig = _sign(body, ts, secret)
    assert not verify_slack_signature(b'{"event":"bye"}', ts, sig, secret)


def test_old_timestamp_rejected():  # noqa: ANN201
    from app.integrations.slack_router import verify_slack_signature

    body = b'{"event":"hi"}'
    ts = str(int(time.time()) - 3600)  # one hour old
    secret = "shh"
    sig = _sign(body, ts, secret)
    assert not verify_slack_signature(body, ts, sig, secret)


def test_no_secret_rejected():  # noqa: ANN201
    from app.integrations.slack_router import verify_slack_signature

    assert not verify_slack_signature(b"x", "0", "v0=deadbeef", "")
