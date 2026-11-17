"""CPU-only regression tests for IQ/Baseten OTLP telemetry and full-shard gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from safetensors.numpy import save_file

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from iq_telemetry import IQTelemetry  # noqa: E402
import baseten_iq_preflight as preflight  # noqa: E402


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.telemetry = IQTelemetry(mode="console", span_exporter=self.exporter)

    def tearDown(self):
        self.assertTrue(self.telemetry.flush())
        self.telemetry.close()

    def test_stage_spans_and_numeric_fields(self):
        with self.telemetry.stage("request", prompt_tokens=3):
            with self.telemetry.stage("prefill", finite=True):
                pass
        spans = self.exporter.get_finished_spans()
        self.assertEqual([s.name for s in spans], ["iq.prefill", "iq.request"])
        self.assertEqual(spans[1].attributes["iq.prompt_tokens"], 3)
        self.assertGreaterEqual(spans[0].attributes["iq.latency_ms"], 0)

    def test_exceptions_do_not_leak_secrets(self):
        with self.assertRaises(ValueError):
            with self.telemetry.stage("checkpoint_load"):
                raise ValueError("token=super-secret and prompt=private-user-text")
        span = self.exporter.get_finished_spans()[-1]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        event = span.events[-1]
        self.assertEqual(event.name, "iq.failure")
        self.assertEqual(event.attributes["error.type"], "ValueError")
        self.assertNotIn("super-secret", str(span.attributes) + str(span.events))
        self.assertNotIn("private-user-text", str(span.attributes) + str(span.events))

    def test_disallow_raw_prompts_and_dynamic_stage_names(self):
        with self.assertRaises(ValueError):
            with self.telemetry.stage("request", prompt_text="private"):
                pass
        with self.assertRaises(ValueError):
            with self.telemetry.stage("model_loader_client_secret"):
                pass

    def test_prod_requires_explicit_secure_collector(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "endpoint"):
                IQTelemetry(mode="otlp")
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT":
                                      "http://public.example.com:4318"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HTTPS"):
                IQTelemetry(mode="otlp")


class TestPreflight(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)
        self.name = "model-mamba-000.safetensors"
        save_file({"weight": np.arange(12, dtype=np.float32).reshape(3, 4)},
                  str(self.root / self.name))
        self.index = {"weight_map": {"weight": self.name}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))
        # This old compiler contract is intentionally NOT GLM-first.
        (self.root / "iq_config.json").write_text(
            json.dumps({"model": {"vocab_size": 128256, "hidden_size": 4096}})
        )
        (self.root / "transplant_manifest.json").write_text(
            json.dumps({"method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
                        "foundation": {"repo": "state-spaces/mamba3-mimo-1.5b"}})
        )
        self.sha = hashlib.sha256((self.root / self.name).read_bytes()).hexdigest()
        self.hash_manifest = self.root / "hashes.json"
        self.hash_manifest.write_text(json.dumps({"shards": {self.name: self.sha}}))
        self.exporter = InMemorySpanExporter()
        self.telemetry = IQTelemetry(mode="console", span_exporter=self.exporter)

    def tearDown(self):
        self.telemetry.flush()
        self.telemetry.close()
        self.work.cleanup()

    def args(self):
        return argparse.Namespace(
            checkpoint=self.root,
            tokenizer_dir=None,
            expected_glm_tokenizer_sha256=None,
            expected_shards=1,
            sha256_manifest=self.hash_manifest,
            telemetry="console",
        )

    def test_header_and_shard_sha256_verification(self):
        weights, shards = preflight.index_inventory(self.root, 1)
        self.assertEqual(preflight.verify_headers(self.root, weights, shards), 1)
        self.assertEqual(preflight.verify_shard_hashes(
            self.root, shards, self.hash_manifest), 1)

    def test_corrupt_header_and_wrong_digest_fail_closed(self):
        with self.assertRaises(ValueError):
            preflight.verify_headers(self.root, {"bad_name": self.name}, [self.name])
        self.hash_manifest.write_text(json.dumps({"shards": {self.name: "0" * 64}}))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            preflight.verify_shard_hashes(
                self.root, [self.name], self.hash_manifest
            )

    def test_legacy_mamba_all256_not_misreported_as_glm(self):
        report = preflight.execute(self.args(), self.telemetry)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["quality_verified"])
        self.assertFalse(report["gpu_launched"])
        self.assertFalse(report["inference_called"])
        self.assertIn("vocab_mismatch", report["findings"])
        spans = self.exporter.get_finished_spans()
        tokenizer = [s for s in spans if s.name == "iq.tokenizer"]
        self.assertEqual(tokenizer[0].status.status_code, StatusCode.ERROR)

    def test_even_passing_metadata_is_not_generation_readiness(self):
        with patch.object(preflight, "audit_foundation",
                          return_value={"status": "metadata_consistent", "findings": []}), \
             patch.object(preflight, "verify_tokenizer_mapping", return_value=True):
            report = preflight.execute(self.args(), self.telemetry)
        self.assertEqual(report["status"], "metadata_consistent_quality_unverified")
        self.assertFalse(report["quality_verified"])
        self.assertFalse(report["gpu_launched"])
        self.assertEqual(report["hashes_verified"], 1)


if __name__ == "__main__":
    unittest.main()
