"""Privacy-conscious OpenTelemetry instrumentation for IQ inference.

Traces contain stage names, physical layer ordinals, shape/dtype/device metadata
and error categories. They NEVER contain token IDs, prompts, tensors or API keys.
OpenTelemetry is optional for model imports. Enable via IQ_OTEL_EXPORTER.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

_CONFIGURED = False


def configure_telemetry(exporter: str | None = None) -> bool:
    """Configure an opt-in tracing exporter: none, console, or otlp.

    Repeated calls are idempotent. No network exporter is enabled by default.
    OTLP requires IQ_OTEL_ENDPOINT, avoiding accidental external disclosure.
    """
    global _CONFIGURED
    selected = (exporter or os.getenv("IQ_OTEL_EXPORTER", "none")).lower()
    if selected == "none":
        return False
    if selected not in ("console", "otlp"):
        raise ValueError("IQ_OTEL_EXPORTER must be none, console, or otlp")
    if _CONFIGURED:
        return True

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    if selected == "otlp":
        endpoint = os.getenv("IQ_OTEL_ENDPOINT")
        if not endpoint:
            raise ValueError("IQ_OTEL_ENDPOINT must be set for OTLP export")
        if not endpoint.startswith(("https://", "http://localhost:", "http://127.0.0.1:")):
            raise ValueError("IQ_OTEL_ENDPOINT must use HTTPS (or local loopback HTTP)")
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter_instance = OTLPSpanExporter(endpoint=endpoint)
    else:
        exporter_instance = ConsoleSpanExporter()

    provider = TracerProvider(resource=Resource.create({"service.name": "iq-inference"}))
    provider.add_span_processor(BatchSpanProcessor(exporter_instance))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True
    return True


@contextmanager
def trace_stage(name: str, **attributes: Any) -> Iterator[None]:
    """Wrap a stage; attribute values must be non-sensitive scalars.

    Does not capture exception messages or stack traces, which could contain
    sensitive prompts or credentials. The stage's exception still propagates.
    """
    if not _CONFIGURED:
        yield
        return

    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    safe = {
        f"iq.{key}": value
        for key, value in attributes.items()
        if isinstance(value, (str, int, float, bool)) and key in {
            "layer_index", "operator", "device", "dtype", "batch_size",
            "sequence_length", "vocab_size", "tensor_count", "shard_count",
            "checkpoint_bytes", "generation_step", "response_model",
        }
    }
    tracer = trace.get_tracer("iq.inference")
    with tracer.start_as_current_span(name, attributes=safe, record_exception=False) as span:
        try:
            yield
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
