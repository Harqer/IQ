"""Contract checks for whole-checkpoint telemetry; NOT a substitute for GPU inference."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class FullIqTelemetrySourceContract(unittest.TestCase):
    def test_python_sources_compile(self):
        for path in [
            "iq_observability/__init__.py",
            "iq_observability/full_trace.py",
            "scripts/test_complete_iq_traced.py",
            "scripts/serve_iq_hf.py",
            "iq_transfer/complete_transplant.py",
        ]:
            with self.subTest(path=path):
                p = ROOT / path
                compile(p.read_text(encoding="utf-8"), str(p), "exec")

    def test_full_checkpoint_not_random_weights(self):
        runner = (ROOT / "scripts/test_complete_iq_traced.py").read_text()
        self.assertIn("load_complete_iq_checkpoint(", runner)
        self.assertIn("recorder.assert_complete()", runner)
        self.assertIn("preflight_iq_checkpoint(", runner)
        self.assertNotIn("initialize_random_module", runner)
        self.assertNotIn("test_only_cpu_reference", runner)

    def test_otel_and_critical_hooks(self):
        trace = (ROOT / "iq_observability/full_trace.py").read_text()
        self.assertIn("OTLPSpanExporter", trace)
        self.assertIn("register_forward_hook", trace)
        self.assertIn("always_call=True", trace)
        self.assertIn("cuda_peak_bytes", trace)
        self.assertIn("expert_assignments", trace)
        self.assertIn('EXPECTED = {"mamba3": 32, "csa": 12, "hca": 12, "moe": 24}', trace)
        self.assertIn("NaN/Inf detected", trace)

    def test_server_and_shard_tracing(self):
        server = (ROOT / "scripts/serve_iq_hf.py").read_text()
        loader = (ROOT / "iq_transfer/complete_transplant.py").read_text()
        self.assertIn('os.getenv("IQ_FULL_TRACE", "0")', server)
        self.assertIn("recorder.assert_complete()", server)
        self.assertIn('X-Baseten-Request-Id', server)
        self.assertIn('telemetry.event("checkpoint.shard.start"', loader)
        self.assertIn('telemetry.event("checkpoint.shard.loaded"', loader)


if __name__ == "__main__":
    unittest.main()
