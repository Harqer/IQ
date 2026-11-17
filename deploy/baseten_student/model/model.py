"""One Baseten prediction interface for architecture-only and injected weights."""
from __future__ import annotations

import os

from iq_cloud.phase_probe import check_phase, load_phase, full_forward_probe


class Model:
    def __init__(self, config=None, **kwargs):
        self.config = config or kwargs.get("config") or {}
        self.phase = check_phase(os.getenv("IQ_PHASE", "architecture_only"))
        self.model = None
        self.tracer = None

    def load(self):
        # Baseten secrets are supplied through the Truss config, not checked into git.
        secret = (self.config.get("secrets") or {}).get("hf_access_token")
        if secret:
            os.environ.setdefault("HF_TOKEN", secret)
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider = TracerProvider(resource=Resource.create({
            "service.name": "iq-glm53-baseten",
            "iq.phase": self.phase,
        }))
        endpoint_set = bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                            or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter() if endpoint_set else ConsoleSpanExporter()
        ))
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("iq.full_student")
        with self.tracer.start_as_current_span("iq.model.load") as span:
            self.model = load_phase(self.phase, glm_root=os.getenv("IQ_GLM_ROOT"))
            span.set_attribute("iq.layers", len(self.model.core.backbone.layers))
            span.set_attribute("iq.phase", self.phase)

    def predict(self, model_input):
        if self.model is None:
            raise RuntimeError("IQ model has not loaded")
        if not isinstance(model_input, dict):
            raise ValueError("request must be a JSON object")
        with self.tracer.start_as_current_span("iq.request") as span:
            try:
                report = full_forward_probe(
                    self.model, model_input.get("prompt"), phase=self.phase,
                    max_new_tokens=model_input.get("max_new_tokens", 4),
                    tracer=self.tracer,
                )
            except Exception as error:
                span.record_exception(error)
                raise
            span.set_attribute("iq.completed_layers", report.completed_layers)
            span.set_attribute("iq.generated_tokens", len(report.generated_ids))
            return {
                "phase": report.phase,
                "weights_injected": self.phase == "injected",
                "architecture": "24 native Mamba-3 MIMO blocks, GLM lexical adapters",
                "tokenizer": "GLM-5.3",
                "input_tokens": report.input_tokens,
                "completed_layers_per_token": report.completed_layers,
                "generated_ids": report.generated_ids,
                "completion": report.output,
                "schema_sha256": report.schema_sha256,
                "tokenizer_sha256": report.tokenizer_sha256,
            }
