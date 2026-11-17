"""No-network tests of the Baseten read-only management inspection client."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import baseten_iq_readonly as client  # noqa: E402


class DummyResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit: int):
        return json.dumps(self.payload).encode("utf-8")


class TestBasetenReadOnly(unittest.TestCase):
    def test_default_sends_no_requests_without_credentials(self):
        output = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(client, "urlopen", side_effect=AssertionError("network used")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(client.main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["requests_sent"], 0)
        self.assertFalse(payload["compute_launched"])
        self.assertEqual(payload["intended_GET_paths"], ["/v1/models"])

    def test_request_is_authenticated_get_only(self):
        recorded = []
        def opener(req, timeout):
            recorded.append((req.get_method(), req.full_url, timeout,
                             req.get_header("Authorization")))
            return DummyResponse({"models": [
                {"id": "abc12345", "name": "IQ", "private_key": "do_not_log"},
            ]})
        data = client.inspect_path("/v1/models", "fake-secret", opener=opener)
        sanitized = client.summarize("/v1/models", data)
        self.assertEqual(recorded, [
            ("GET", "https://api.baseten.co/v1/models", 10,
             "Api-Key fake-secret"),
        ])
        self.assertNotIn("do_not_log", json.dumps(sanitized))
        self.assertNotIn("fake-secret", json.dumps(sanitized))
        self.assertEqual(sanitized["models"][0]["name"], "IQ")

    def test_rejects_injection_and_unsupported_actions(self):
        for value in ("../../secrets", "valid/activate", "x?delete=true", ""):
            with self.assertRaises(ValueError):
                client.paths(value)
        for value in ("/v1/models/x/deployments/x/promote", "/v1/models/x/predict",
                      "/v1/secrets", "https://evil.example/"):
            with self.assertRaises(ValueError):
                client.inspect_path(value, "fake-secret",
                    opener=lambda *_args, **_kwargs: None)

    def test_bad_api_response_does_not_leak_credentials(self):
        from urllib.error import HTTPError
        def opener(_req, timeout):
            raise HTTPError("https://api.baseten.co/v1/models", 401,
                            "Authorization: Api-Key secret-contents", None, None)
        with self.assertRaisesRegex(RuntimeError, "401") as cm:
            client.inspect_path("/v1/models", "secret-contents", opener=opener)
        self.assertNotIn("secret-contents", str(cm.exception))

    def test_inspect_requires_credential(self):
        err = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), contextlib.redirect_stderr(err):
            self.assertEqual(client.main(["--inspect"]), 2)
        self.assertIn("blocked", err.getvalue())


if __name__ == "__main__":
    unittest.main()
