"""Baseten IQ235: architecture-only default; opt-in, strictly gated source probe."""
from __future__ import annotations
from os import environ

from iq_cloud.full_iq235 import load_phase_a, load_pretrained_source, generate_probe


class Model:
    def __init__(self, **kwargs):
        self.config = kwargs.get("config", {})
        self.state = None
        self.tracer = None

    def load(self):
        phase = environ.get("IQ_PHASE")
        if phase not in {"architecture_only", "pretrained_source"}:
            raise RuntimeError("IQ_PHASE must explicitly select an allowed mode")
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=Resource.create({
            "service.name": "iq235-baseten",
            "iq.phase": phase,
            "iq.checkpoint_source": "kofecheckmate/iq-glm53-all256",
        }))
        if environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or environ.get(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        ):
            exporter = OTLPSpanExporter()
        else:
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("iq235.full")
        try:
            with self.tracer.start_as_current_span(
                "iq235.load.full_architecture" if phase == "architecture_only"
                else "iq235.load.pretrained_source"
            ):
                token = (self.config.get("secrets") or {}).get("hf_access_token")
                if token:
                    environ.setdefault("HF_TOKEN", token)
                self.state = (load_phase_a() if phase == "architecture_only"
                              else load_pretrained_source(tracer=self.tracer))
        except Exception:
            # A failing model initialization may terminate the worker before
            # the batch processor exports its span; flush before propagation.
            provider.force_flush(timeout_millis=5000)
            raise

    def predict(self, model_input):
        if self.state is None:
            raise RuntimeError("full IQ architecture has not been loaded")
        if not isinstance(model_input, dict):
            raise ValueError("predict request must be a JSON object")
        with self.tracer.start_as_current_span("iq235.inference.full"):
            return generate_probe(
                self.state,
                model_input.get("prompt", "1 2 3"),
                max_new_tokens=model_input.get("max_new_tokens", 1),
                tracer=self.tracer,
            )
