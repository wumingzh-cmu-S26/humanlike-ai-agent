# =============================================================================
# Multi-stage build: smaller final image, faster CI cache.
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/app/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libssl3 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 app

USER app
WORKDIR /home/app

COPY --from=builder --chown=app:app /root/.local /home/app/.local
COPY --chown=app:app app ./app
COPY --chown=app:app personalities ./personalities
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
