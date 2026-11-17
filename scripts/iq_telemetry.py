"""OpenTelemetry for IQ preflight and future Baseten serving boundaries.

No model, torch, network, CUDA or Baseten imports. No prompts, completions, token
IDs, checkpoint paths, secrets or raw exception messages are ever recorded.
"""
from __future__ import annotations

import os
import re
import sys
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor,
)
from opentelemetry.trace import Status, StatusCode

STAGES = frozenset({
    "preflight", "metadata", "tokenizer", "shard_inventory", "tensor_headers",
    "tensor_hashes", "tensor_validation", "transfer_validation",
    "checkpoint_load", "donor_load", "adapter_load", "model_load",
    "request", "tokenize", "prefill", "first_token", "decode",
    "postprocess", "response",
})
# Low-cardinality, numeric/boolean, non-secret attributes only.
ATTRIBUTES = frozenset({
    "shards", "indexed_tensors", "verified_tensors", "bytes",
    "prompt_tokens", "completion_tokens", "max_new_tokens",
    "model_layers", "donor_layers", "gpu_allocated_bytes",
    "gpu_reserved_bytes", "nan_count", "inf_count", "finite",
    "tokenizer_verified", "transfer_verified", "hashes_verified",
    "checkpoint_verified", "success", "http_status",
    "decode_steps", "latency_ms", "time_to_first_token_ms",
})
_STAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


def _validate_otlp_target() -> None:
    value = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
             or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
    if not value:
        raise RuntimeError("OTLP tracing endpoint is not configured")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("Invalid OTLP tracing endpoint")
    if parsed.username or parsed.password or parsed.fragment:
        raise RuntimeError("OTLP URL must not contain embedded credentials or fragments")
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost", "127.0.0.1", "::1",
    }:
        raise RuntimeError("Remote OTLP tracing requires HTTPS")


class IQTelemetry:
    """Explicit exporter selection; production OTLP never silently falls back."""

    def __init__(self, mode: str = "console", *, span_exporter=None) -> None:
        if mode not in {"console", "otlp"}:
            raise ValueError("Telemetry mode must be 'console' or 'otlp'")
        if mode == "otlp":
            _validate_otlp_target()
        resource = Resource.create({
            "service.name": "iq-baseten",
            "deployment.environment.name": (
                "baseten" if mode == "otlp" else "offline-preflight"
            ),
        })
        self.provider = TracerProvider(resource=resource)
        if span_exporter is not None:
            exporter = span_exporter
            processor = SimpleSpanProcessor(exporter)
        elif mode == "console":
            exporter = ConsoleSpanExporter(out=sys.stderr)
            processor = SimpleSpanProcessor(exporter)
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter()
            processor = BatchSpanProcessor(exporter, max_queue_size=1024,
                                           max_export_batch_size=128)
        self.provider.add_span_processor(processor)
        self.tracer = self.provider.get_tracer("iq.telemetry")
        self.mode = mode

        # Metrics use OTLP only when a metrics endpoint (or a general endpoint)
        # is explicitly configured. No implicit traffic to localhost.
        self.meter_provider = None
        self.failure_counter = None
        self.stage_duration = None
        if mode == "otlp" and (
            os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        ):
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(), export_interval_millis=30000
            )
            self.meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            meter = self.meter_provider.get_meter("iq.telemetry")
            self.failure_counter = meter.create_counter(
                "iq_stage_failures", unit="1",
                description="Number of failed IQ stages",
            )
            self.stage_duration = meter.create_histogram(
                "iq_stage_duration", unit="ms",
                description="Elapsed time per IQ stage",
            )

    @contextmanager
    def stage(self, name: str, **attributes):
        """Record a safe operation boundary, numeric attributes and failure type."""
        if name not in STAGES or not _STAGE_PATTERN.fullmatch(name):
            raise ValueError("Unrecognized IQ tracing stage")
        for key, value in attributes.items():
            if key not in ATTRIBUTES or type(value) not in {int, float, bool}:
                raise ValueError("Telemetry attributes must be approved numeric/boolean fields")
        started = time.monotonic()
        with self.tracer.start_as_current_span(
            "iq." + name, record_exception=False, set_status_on_exception=False
        ) as span:
            for key, value in attributes.items():
                span.set_attribute("iq." + key, value)
            try:
                yield span
            except Exception as exc:
                # Deliberately no record_exception(): it can leak paths, prompt
                # contents, credentials, or arbitrary user-controlled text.
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("iq.failure", {"error.type": type(exc).__name__})
                if self.failure_counter is not None:
                    self.failure_counter.add(1, {"stage": name})
                raise
            finally:
                elapsed = (time.monotonic() - started) * 1000
                span.set_attribute("iq.latency_ms", round(elapsed, 3))
                if self.stage_duration is not None:
                    self.stage_duration.record(elapsed, {"stage": name})

    @staticmethod
    def annotate(span, **attributes) -> None:
        """Allow only low-cardinality numeric/boolean fields on the current span."""
        for key, value in attributes.items():
            if key not in ATTRIBUTES or type(value) not in {int, float, bool}:
                raise ValueError("Unapproved telemetry attribute")
            span.set_attribute("iq." + key, value)

    def flush(self, timeout_ms: int = 5000) -> bool:
        traces_ok = bool(self.provider.force_flush(timeout_millis=timeout_ms))
        metrics_ok = (self.meter_provider is None or
                      bool(self.meter_provider.force_flush(timeout_millis=timeout_ms)))
        return traces_ok and metrics_ok

    def close(self) -> None:
        self.provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()
