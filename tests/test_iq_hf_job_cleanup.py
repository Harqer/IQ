"""CPU-only smoke tests: prove the paid HF Job is canceled on success and failure."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "iq_hf_inference.py"
spec = importlib.util.spec_from_file_location("iq_hf_inference", SCRIPT)
assert spec and spec.loader
iq = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = iq
spec.loader.exec_module(iq)


class FakeCompletion:
    choices = [SimpleNamespace(message=SimpleNamespace(content="4"))]


class FakeInferenceClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: FakeCompletion())
        )


class TestHFJobCleanup(unittest.TestCase):
    def setup_mocks(self, *, ready: bool):
        self.patches = [
            patch.object(iq, "whoami"),
            patch.object(iq, "get_token", return_value="test-token"),
            patch.object(iq, "get_deployed_commit", return_value="a" * 40),
            patch.object(iq, "run_job", return_value=SimpleNamespace(
                id="a" * 24, url="https://example.invalid/job"
            )),
            patch.object(iq, "cancel_job"),
            patch.object(iq, "InferenceClient", FakeInferenceClient),
            patch.object(iq, "wait_ready", side_effect=None if ready else RuntimeError("not ready"),
                         return_value="https://example.invalid"),
        ]
        return [p.start() for p in self.patches]

    def tearDown(self):
        for p in getattr(self, "patches", []):
            p.stop()

    def test_success_cancels_job(self):
        mocks = self.setup_mocks(ready=True)
        args = SimpleNamespace(namespace="kofecheckmate", interactive=False,
                               prompt="2+2", max_tokens=8, ready_timeout=10,
                               request_timeout=20)
        iq.run(args)
        mocks[4].assert_called_once_with(job_id="a" * 24, namespace="kofecheckmate")

    def test_failed_readiness_cancels_job(self):
        mocks = self.setup_mocks(ready=False)
        args = SimpleNamespace(namespace="kofecheckmate", interactive=False,
                               prompt="2+2", max_tokens=8, ready_timeout=10,
                               request_timeout=20)
        with self.assertRaises(RuntimeError):
            iq.run(args)
        mocks[4].assert_called_once_with(job_id="a" * 24, namespace="kofecheckmate")


if __name__ == "__main__":
    unittest.main()
