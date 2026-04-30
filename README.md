# Human-Like AI Agent

Production-grade conversational AI with personality, memory, hybrid RAG, multi-modal output, and multi-channel deployment (web, Telegram, Slack, digital human).

Built with **FastAPI**, **LangChain + OpenAI function calling**, **BM25 + FAISS + Chroma + cross-encoder rerank**, **Azure TTS**, **Google Calendar/Tasks**, **OpenTelemetry**, and deployed to **AWS ECS Fargate** / **Kubernetes**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Channels                                       │
│   Web Chat (SSE)    Telegram Bot     Slack Bot     Digital Human     │
└────────┬───────────────┬─────────────────┬─────────────────┬─────────┘
         │               │                 │                 │
         ▼               ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  FastAPI Gateway                                      │
│   JWT auth · Rate limit · Circuit breakers · OTel · Structured logs  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Agent Orchestrator (LangChain)                       │
│   Personality YAML · Sentiment perception · OpenAI function calling   │
└─────┬───────────────┬───────────────┬────────────────┬───────────────┘
      │               │               │                │
      ▼               ▼               ▼                ▼
   Memory          Hybrid RAG       Tools           Voice / Avatar
   ─ Short-term    ─ BM25           ─ rag_search    ─ Azure TTS
   ─ Summary       ─ FAISS (dense)  ─ web_search    ─ Viseme stream
   ─ Long-term     ─ Chroma         ─ time          ─ Emotion timeline
     vector mem    ─ RRF fusion     ─ Calendar
                   ─ Cross-encoder  ─ Tasks
                     rerank         ─ Telegram/Slack send
```

---

## Quick start

```bash
# 1. Clone & set up
cp .env.example .env             # fill in OPENAI_API_KEY at minimum
pip install -r requirements.txt
pip install -e ".[dev]"

# 2. Seed sample docs
python scripts/seed_documents.py

# 3. Run
uvicorn app.main:app --reload
# → http://localhost:8000/docs

# 4. Get a JWT
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo"}'

# 5. Chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"What can you do?"}'
```

### Docker
```bash
docker compose up --build
```

---

## Endpoints

| Method | Path                              | Purpose                          |
| ------ | --------------------------------- | -------------------------------- |
| POST   | `/auth/token`                     | Issue JWT                        |
| POST   | `/chat`                           | Synchronous chat                 |
| POST   | `/chat/stream`                    | SSE token streaming              |
| POST   | `/rag/ingest`                     | Add documents to the KB          |
| POST   | `/rag/search`                     | Hybrid retrieval                 |
| POST   | `/voice/tts`                      | Synthesize speech (mp3)          |
| POST   | `/voice/digital-human`            | TTS + viseme/emotion event stream|
| GET    | `/integrations/google/connect`    | Start Google OAuth               |
| GET    | `/integrations/google/callback`   | Finish Google OAuth              |
| POST   | `/integrations/telegram/webhook`  | Telegram inbound                 |
| POST   | `/integrations/slack/events`      | Slack inbound (signature verified)|
| GET    | `/healthz` / `/readyz`            | Health & readiness               |

---

## Hybrid RAG

Three retrievers run in parallel, then are fused with **Reciprocal Rank Fusion** and reranked with a **cross-encoder**:

| Retriever       | Strength                                        |
| --------------- | ----------------------------------------------- |
| BM25            | Lexical / keyword matches, no embedding needed  |
| FAISS (dense)   | Fast in-memory cosine over OpenAI embeddings    |
| Chroma          | Persistent, metadata-filterable                 |
| Cross-encoder   | `ms-marco-MiniLM-L-6-v2` — final precision pass |

Benchmark with:
```bash
python scripts/benchmark_rag.py --num-docs 1000 --num-queries 50
```

---

## Memory

| Tier        | Backed by                       | Purpose                          |
| ----------- | ------------------------------- | -------------------------------- |
| Short-term  | In-process sliding window       | Last N turns                     |
| Summary     | LLM-generated rolling summary   | Compress older history           |
| Long-term   | Chroma vector store, per-session| Semantic recall across sessions  |

---

## Personalities

Drop a YAML file in `personalities/`:

```yaml
name: companion
voice: en-US-JennyNeural
traits: { warmth: 0.9, formality: 0.2 }
system_prompt: |
  You are Aria, a warm and attentive AI companion...
sentiment_responses:
  negative: "I can hear that's weighing on you. "
  positive: "Love that energy. "
```

Three are shipped: `companion`, `professional`, `playful`.

---

## Observability

- **OpenTelemetry** traces auto-instrument FastAPI + httpx; OTLP gRPC exporter.
- **Structured JSON logs** (structlog) in production, console in dev.
- **CloudWatch dashboard** JSON in `infra/otel/cloudwatch_dashboard.json` — request rate, p50/p95/p99 latency, breaker state, RAG retrieval latency, ECS CPU/mem, error log query.
- **Circuit breakers** (pybreaker) wrap every external call: OpenAI, Google, Telegram, Slack, Azure.
- **Rate limiting** (slowapi) per JWT or IP.

---

## Deployment

- **Docker** — multi-stage, non-root, healthcheck, ~600 MB image.
- **AWS ECS Fargate** — `infra/ecs/task-definition.json` with sidecar OTel collector and Secrets Manager wiring.
- **Kubernetes** — `infra/k8s/deployment.yaml` with HPA scaling on CPU+memory, rolling updates, readiness/liveness probes.
- **GitHub Actions** — `.github/workflows/ci.yml` runs lint → mypy → pytest, then builds and pushes to GHCR on `main`.

---

## Testing

```bash
pytest -q
```

Covers: JWT round-trip, login, RRF fusion, BM25 ranking, circuit-breaker state machine, Slack signature verification (valid / tampered body / old timestamp / no secret), personality registry, short-term memory windowing, tool registry presence, health endpoints.

---

## Project structure

```
app/
  api/            FastAPI routers (auth, chat, rag, voice, health) + middleware + schemas + errors
  agents/         Orchestrator, personality registry
  core/           Config, logging, security (JWT), circuit breakers, rate limit, observability
  memory/         Short-term, summary, long-term vector
  rag/            Embeddings, BM25, FAISS, Chroma, RRF, cross-encoder rerank, hybrid retriever, chunker
  perception/     Sentiment analysis (DistilBERT SST-2 + lexicon fallback)
  tools/          rag_search · web_search · time · calendar · tasks · telegram · slack
  voice/          Azure TTS · voice-clone hook
  digital_human/  Viseme + emotion event stream builder
  integrations/   Google OAuth · Telegram webhook · Slack events (signed)
infra/
  ecs/            Fargate task definition
  k8s/            Deployment, HPA, Service, Secret example
  otel/           Collector config, CloudWatch dashboard JSON
personalities/    companion.yaml · professional.yaml · playful.yaml
scripts/          seed_documents.py · benchmark_rag.py
tests/            8 test modules
```

---

## Resume bullet mapping

| Resume claim                                                                              | Where it lives                                                                                                |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| LangChain conversational AI with OpenAI function calling                                  | `app/agents/orchestrator.py` — `create_openai_functions_agent` + `AgentExecutor`                              |
| Hybrid RAG: BM25 + FAISS + Chroma + cross-encoder rerank                                  | `app/rag/{bm25_retriever,faiss_retriever,chroma_retriever,reranker,fusion,hybrid_retriever}.py`               |
| Reciprocal Rank Fusion                                                                    | `app/rag/fusion.py`                                                                                           |
| Multi-tier memory (short-term, summary, long-term vector)                                 | `app/memory/{short_term,summary,long_term,manager}.py`                                                        |
| FastAPI async backend with JWT, Pydantic v2, circuit breakers, rate limiting              | `app/main.py`, `app/core/{security,circuit_breaker,rate_limit}.py`, `app/api/`                                |
| SSE streaming chat                                                                        | `app/api/chat.py` → `/chat/stream`                                                                            |
| Sentiment-aware perception                                                                | `app/perception/sentiment.py` (DistilBERT SST-2 + lexicon fallback)                                            |
| Personality engine with YAML configs                                                      | `app/agents/personality.py` + `personalities/*.yaml`                                                          |
| Google Calendar & Tasks via OAuth 2.0                                                     | `app/integrations/google_oauth.py` + `app/tools/google_{calendar,tasks}_tool.py`                              |
| Telegram bot (python-telegram-bot v20) with secret-token webhook                          | `app/integrations/telegram_router.py`                                                                         |
| Slack bot (slack-bolt/sdk) with HMAC signature verification + tenacity retries            | `app/integrations/slack_router.py`                                                                            |
| Azure TTS with viseme + word-boundary capture                                             | `app/voice/tts.py`                                                                                            |
| Digital-human event stream (visemes + emotion + captions)                                 | `app/digital_human/event_stream.py`                                                                           |
| Voice cloning hook                                                                        | `app/voice/voice_clone.py`                                                                                    |
| OpenTelemetry tracing                                                                     | `app/core/observability.py` + `infra/otel/otel-collector-config.yaml`                                          |
| CloudWatch dashboards                                                                     | `infra/otel/cloudwatch_dashboard.json`                                                                        |
| Docker multi-stage build, ECS Fargate, K8s with HPA                                       | `Dockerfile`, `docker-compose.yml`, `infra/ecs/task-definition.json`, `infra/k8s/deployment.yaml`              |
| GitHub Actions CI: lint → type → test → build → push                                      | `.github/workflows/ci.yml`                                                                                    |
| Latency benchmarks                                                                        | `scripts/benchmark_rag.py`                                                                                    |

---

## License
MIT.
