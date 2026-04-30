"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app import __version__
from app.api.auth import router as auth_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestContextMiddleware
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import limiter

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    log.info("app_startup", version=__version__)
    # Lazy imports — heavy ML deps loaded only on first use.
    yield
    log.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Human-Like AI Agent",
        version=__version__,
        description="Production-grade conversational AI with personality, memory, and multi-modal capabilities.",
        lifespan=lifespan,
    )

    # ---- Rate limiting ----
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda r, e: None)  # overridden below
    app.add_middleware(SlowAPIMiddleware)

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request context / access log ----
    app.add_middleware(RequestContextMiddleware)

    # ---- Routers ----
    app.include_router(health_router)
    app.include_router(auth_router)

    from app.api.chat import router as chat_router
    from app.api.rag import router as rag_router
    from app.api.voice import router as voice_router
    from app.integrations.google_oauth import router as google_router
    from app.integrations.slack_router import router as slack_router
    from app.integrations.telegram_router import router as telegram_router

    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(voice_router)
    app.include_router(google_router)
    app.include_router(telegram_router)
    app.include_router(slack_router)

    # ---- Errors ----
    register_exception_handlers(app)

    # ---- Observability ----
    from app.core.observability import setup_otel

    setup_otel(app)

    return app


app = create_app()
