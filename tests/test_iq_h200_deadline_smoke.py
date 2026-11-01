"""CPU-only deadline, failure-classification and localhost SSE smoke tests."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import BytesIO, StringIO
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from scripts import iq_hf_full78_cli as cli
from scripts.h200_full78_smoke import read_smoke_stream


class H200DeadlineTests(unittest.TestCase):
    def test_deadline_is_based_on_real_hf_job_creation(self):
        job = SimpleNamespace(created_at="2026-11-01T09:15:36.954Z")
        expected = datetime(2026, 11, 1, 9, 15, 36, 954000, timezone.utc).timestamp() + 1800
        self.assertEqual(cli.job_deadline_epoch(job), expected)

    def test_deadline_rejects_naive_timestamp(self):
        with self.assertRaisesRegex(RuntimeError, "timezone"):
            cli.job_deadline_epoch(SimpleNamespace(created_at="2026-11-01T09:15:36"))

    def test_request_timeout_is_bounded_by_remaining_job_lifetime(self):
        with patch("scripts.iq_hf_full78_cli.time.time", return_value=1000):
            self.assertEqual(cli.check_request_budget(1500, 1200), 410)
            self.assertEqual(cli.check_request_budget(1500, 30), 30)
            with self.assertRaisesRegex(TimeoutError, "expires too soon"):
                cli.check_request_budget(1090, 1200)

    def test_running_proxy_failure_distinguished_from_timeout(self):
        job = SimpleNamespace(status=SimpleNamespace(stage="RUNNING"))
        with patch("scripts.iq_hf_full78_cli.inspect_job", return_value=job), \
             patch("scripts.iq_hf_full78_cli.time.time", return_value=1000):
            result = cli.classify_request_error("a" * 24, 1450, ValueError("Bad Request"))
            self.assertIn("ENDPOINT_OR_PROXY_ERROR_WHILE_RUNNING", str(result))
            self.assertIn("remaining_seconds=450", str(result))

    def test_terminal_and_deadline_are_distinct(self):
        job = SimpleNamespace(status=SimpleNamespace(stage="CANCELED"))
        with patch("scripts.iq_hf_full78_cli.inspect_job", return_value=job), \
             patch("scripts.iq_hf_full78_cli.time.time", return_value=1000):
            self.assertIn("REMOTE_JOB_STOPPED_CANCELED",
                          str(cli.classify_request_error("a" * 24, 1500, OSError("closed"))))
            self.assertIn("JOB_DEADLINE_EXPIRED_OR_IMMINENT",
                          str(cli.classify_request_error("a" * 24, 1080, OSError("closed"))))


class H200SmokeSSETests(unittest.TestCase):
    def test_sse_stream_requires_actual_terminal_finish_and_done(self):
        event = b'data: {"choices":[{"delta":{"content":"4"},"finish_reason":null}]}\n\n'
        finish = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        stream = BytesIO(event + finish + b"data: [DONE]\n\n")
        with redirect_stdout(StringIO()):
            answer, reason, count = read_smoke_stream(stream)
        self.assertEqual((answer, reason, count), ("4", "stop", 2))

    def test_truncated_stream_is_never_reported_success(self):
        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                read_smoke_stream(BytesIO(
                    b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
                ))


if __name__ == "__main__":
    unittest.main()
