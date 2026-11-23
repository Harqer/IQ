"""Safety and OTel integration smoke tests; no GPU or API credentials."""
from __future__ import annotations

import unittest

import iq_telemetry


class TraceNoopTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(iq_telemetry.configure_telemetry("none"))

    def test_stage_preserves_exception_without_tracing(self):
        try:
            with iq_telemetry.trace_stage("iq.shard.load", layer_index=5):
                raise ValueError("expected")
        except ValueError as e:
            self.assertEqual(str(e), "expected")
        else:
            self.fail("exception must propagate")

    def test_unknown_exporter_fails_closed(self):
        with self.assertRaises(ValueError):
            iq_telemetry.configure_telemetry("not-a-supported-exporter")


class InMemoryOTelTests(unittest.TestCase):
    def test_sanitized_attributes_and_error_category(self):
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:
            self.skipTest("OpenTelemetry SDK not installed")
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # Use explicit provider without setting a process-global tracer.
        original = trace.get_tracer
        original_configured = iq_telemetry._CONFIGURED
        try:
            trace.get_tracer = lambda *args, **kwargs: provider.get_tracer("iq.test")
            iq_telemetry._CONFIGURED = True
            with self.assertRaises(ValueError):
                with iq_telemetry.trace_stage(
                    "iq.layer",
                    layer_index=3,
                    operator="Mamba3",
                    secret_key="MUST_NOT_APPEAR",
                    prompt="MUST_NOT_APPEAR",
                ):
                    raise ValueError("PROMPT_OR_API_KEY_SHOULD_NOT_LEAK")
        finally:
            iq_telemetry._CONFIGURED = original_configured
            trace.get_tracer = original
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes["iq.layer_index"], 3)
        self.assertEqual(spans[0].attributes["error.type"], "ValueError")
        self.assertNotIn("secret_key", str(spans[0].attributes))
        self.assertNotIn("MUST_NOT_APPEAR", str(spans[0]))
        self.assertNotIn("PROMPT_OR_API_KEY", str(spans[0]))


if __name__ == "__main__":
    unittest.main()
