"""OpenTelemetry tracing setup."""
from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


def setup_otel(app: Any) -> None:
    settings = get_settings()
    if not settings.otel_exporter_otlp_endpoint:
        log.info("otel_disabled", reason="no_endpoint_configured")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBasedTraceIdRatio(settings.otel_traces_sampler_arg),
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
            )
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        log.info("otel_enabled", endpoint=settings.otel_exporter_otlp_endpoint)
    except Exception as e:
        log.warning("otel_setup_failed", error=str(e))


def tracer(name: str = "humanlike-ai-agent"):  # noqa: ANN201
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:  # pragma: no cover

        class _Noop:
            def start_as_current_span(self, *_a, **_kw):  # noqa: ANN201
                from contextlib import nullcontext

                return nullcontext()

        return _Noop()
