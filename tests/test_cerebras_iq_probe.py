"""Offline regression tests: no credentials, API calls, GPUs, or model weights."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.cerebras_iq_probe import run_probe


class FakeModels:
    def __init__(self, ids=None, failure=None):
        self.ids = ids if ids is not None else ["public-model", "test-endpoint"]
        self.failure = failure
        self.calls = 0

    def list(self):
        self.calls += 1
        if self.failure:
            raise self.failure
        return SimpleNamespace(data=[SimpleNamespace(id=x) for x in self.ids])


class FakeCompletions:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return SimpleNamespace(model=kwargs["model"], choices=[SimpleNamespace(message=SimpleNamespace(content="IQ_OK"))])


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.models = FakeModels()
        self.completions = FakeCompletions()
        self.client = SimpleNamespace(
            models=self.models,
            chat=SimpleNamespace(completions=self.completions),
        )

    def test_default_checks_account_without_inference(self):
        report = run_probe(self.client)
        self.assertEqual(report["status"], "api_connected")
        self.assertTrue(report["api_authenticated"])
        self.assertFalse(report["iq_inference_executed"])
        self.assertEqual(self.completions.calls, [])

    def test_fails_closed_without_endpoint(self):
        report = run_probe(self.client, execute_iq=True, confirm_deployed=True, allow_billed_request=True)
        self.assertEqual(report["status"], "blocked_no_iq_endpoint")
        self.assertEqual(self.completions.calls, [])

    def test_fails_closed_without_deployment_confirmation(self):
        report = run_probe(self.client, endpoint_id="test-endpoint", execute_iq=True, allow_billed_request=True)
        self.assertEqual(report["status"], "blocked_unverified_endpoint")
        self.assertEqual(self.completions.calls, [])

    def test_fails_closed_without_billing_confirmation(self):
        report = run_probe(self.client, endpoint_id="test-endpoint", execute_iq=True, confirm_deployed=True)
        self.assertEqual(report["status"], "blocked_billing_not_approved")
        self.assertEqual(self.completions.calls, [])

    def test_single_request_explicitly_approved(self):
        report = run_probe(
            self.client,
            endpoint_id="test-endpoint",
            execute_iq=True,
            confirm_deployed=True,
            allow_billed_request=True,
        )
        self.assertEqual(report["status"], "dedicated_endpoint_responded")
        self.assertEqual(len(self.completions.calls), 1)
        self.assertEqual(self.completions.calls[0]["model"], "test-endpoint")
        self.assertEqual(self.completions.calls[0]["max_completion_tokens"], 8)
        self.assertFalse(report["checkpoint_weights_verified_by_cerebras"])

    def test_key_api_error_sanitized(self):
        self.models.failure = ValueError("secret-key-would-be-here")
        report = run_probe(self.client)
        self.assertEqual(report["status"], "api_error")
        self.assertNotIn("secret-key", str(report))
        self.assertEqual(self.completions.calls, [])

    def test_completion_error_sanitized(self):
        self.completions.failure = ValueError("secret-key-in-remote-error")
        report = run_probe(
            self.client,
            endpoint_id="test-endpoint",
            execute_iq=True,
            confirm_deployed=True,
            allow_billed_request=True,
        )
        self.assertEqual(report["status"], "dedicated_inference_error")
        self.assertNotIn("secret-key", str(report))

    def test_public_model_not_accidentally_used(self):
        report = run_probe(
            self.client,
            endpoint_id="public-model",
            execute_iq=True,
            allow_billed_request=True,
        )
        self.assertEqual(report["status"], "blocked_unverified_endpoint")
        self.assertEqual(self.completions.calls, [])


if __name__ == "__main__":
    unittest.main()
