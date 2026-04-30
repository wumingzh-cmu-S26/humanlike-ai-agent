"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application ----
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ---- JWT ----
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # ---- OpenAI ----
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2048

    # ---- RAG ----
    chroma_persist_dir: str = "./data/chroma"
    faiss_index_path: str = "./data/faiss.index"
    rag_top_k: int = 8
    rag_rerank_top_k: int = 4
    rag_rrf_k: int = 60
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Memory ----
    memory_window_size: int = 10
    memory_summary_token_limit: int = 2000
    long_term_memory_top_k: int = 5
    redis_url: str = "redis://localhost:6379/0"

    # ---- Personality ----
    default_personality: str = "companion"
    personality_dir: str = "./personalities"

    # ---- Telegram ----
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = ""
    telegram_webhook_url: str = ""

    # ---- Slack ----
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_app_token: str = ""

    # ---- Google ----
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/integrations/google/callback"
    google_token_dir: str = "./data/google_tokens"

    # ---- Azure TTS ----
    azure_speech_key: str = ""
    azure_speech_region: str = "eastus"
    azure_speech_voice: str = "en-US-JennyNeural"

    # ---- Circuit breaker ----
    circuit_breaker_fail_max: int = 5
    circuit_breaker_reset_timeout: int = 60

    # ---- Rate limiting ----
    rate_limit_per_minute: int = 60

    # ---- Observability ----
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "humanlike-ai-agent"
    otel_traces_sampler_arg: float = 1.0

    # ---- CORS ----
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    def ensure_dirs(self) -> None:
        for path in (
            self.chroma_persist_dir,
            self.google_token_dir,
            Path(self.faiss_index_path).parent,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
