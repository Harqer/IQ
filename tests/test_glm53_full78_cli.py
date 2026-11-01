"""CPU-only CLI safety tests: never submit an H200 or contact Hugging Face."""
from __future__ import annotations

import io
import stat
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import iq_hf_full78_cli as cli


class CLITests(unittest.TestCase):
    def test_job_command_uses_full78_server_not_mocha(self):
        command = cli.command_for("a" * 40)
        self.assertEqual(command[:2], ["bash", "-lc"])
        self.assertIn("scripts.serve_iq_full78_hf", command[-1])
        self.assertNotIn("scripts/serve_iq_hf.py", command[-1])
        self.assertNotIn("/mocha", command[-1])
        self.assertIn("git checkout -q " + "a" * 40, command[-1])

    def test_hf_job_kwargs_match_installed_sdk_before_billing(self):
        # Regression for HfApi.run_job() unexpected attempts kwarg.
        options = cli.job_options("a" * 40)
        self.assertNotIn("attempts", options)
        self.assertEqual(options["flavor"], "h200")
        self.assertEqual(options["timeout"], "30m")
        self.assertEqual(options["expose"], [8000])
        self.assertTrue(all(v.read_only for v in options["volumes"]))
        cli.validate_jobs_api(options)

    def test_sdk_mismatch_rejected_without_submitting_job(self):
        def old_api(*, image, command):
            return None
        options = cli.job_options("a" * 40)
        with patch.object(cli, "run_job", old_api):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                cli.validate_jobs_api(options)

    def test_hf_job_log_stream_is_supported_before_billing(self):
        cli.validate_logs_api()
        def incompatible(*, job_id):
            return iter(())
        with patch.object(cli, "fetch_job_logs", incompatible):
            with self.assertRaisesRegex(RuntimeError, "fetch_job_logs"):
                cli.validate_logs_api()

    def test_remote_job_log_stream_prints_and_saves_every_line(self):
        with tempfile.TemporaryDirectory() as temp:
            job_id = "b" * 24
            path = Path(temp) / "job.log"
            failure = []
            with patch.object(cli, "fetch_job_logs", return_value=iter([
                "IQ_HF_FULL78_BASE_SHARD_STAGED=1/315\n",
                "IQ_HF_FULL78_BASE_SHARD_LOADED=1/315\n",
            ])) as fetch:
                with patch("builtins.print"):
                    cli.follow_job_logs(job_id, path, failure)
            fetch.assert_called_once_with(
                job_id=job_id, namespace=cli.EXPECTED_ACCOUNT,
                follow=True, tail=100,
            )
            self.assertEqual(failure, [])
            self.assertEqual(path.read_text().splitlines(), [
                "IQ_HF_FULL78_BASE_SHARD_STAGED=1/315",
                "IQ_HF_FULL78_BASE_SHARD_LOADED=1/315",
            ])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_remote_log_failure_reports_error_and_readiness_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "job.log"
            failures = []
            with patch.object(cli, "fetch_job_logs",
                              side_effect=RuntimeError("remote logs unavailable")):
                with patch("builtins.print"):
                    cli.follow_job_logs("b" * 24, path, failures)
            self.assertTrue(failures)
            with self.assertRaisesRegex(RuntimeError, "log stream failed"):
                cli.wait_for_server("b" * 24, "token", 1, failures)

    def test_credential_must_match_checkpoint_owner(self):
        with patch.object(cli, "get_token", return_value="secret"):
            with patch.object(cli, "whoami", return_value={"name": "Visionboxx"}):
                with self.assertRaisesRegex(RuntimeError, "wrong account|expected"):
                    cli.validate_auth()

    def test_chat_dialogue_and_cleanup_with_fake_client(self):
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="amber"))
        ]
        with patch.object(cli, "InferenceClient", return_value=fake):
            with patch("builtins.input", side_effect=["Remember amber", "What was the word?", "/quit"]):
                cli.chat_loop(
                    "https://example.hf.jobs", "secret", max_tokens=2, timeout=1,
                )
        self.assertEqual(fake.chat.completions.create.call_count, 2)
        h = fake.chat.completions.create.call_args_list[-1].kwargs["messages"]
        self.assertEqual([x["role"] for x in h], ["user", "assistant", "user"])
        self.assertEqual(h[0]["content"], "Remember amber")

    def test_successful_cancellation_verified(self):
        response = MagicMock(status=MagicMock(stage="CANCELED"))
        with patch.object(cli, "cancel_job") as cancel:
            with patch.object(cli, "inspect_job", return_value=response):
                self.assertTrue(cli.cancel_and_verify("a" * 24))
                cancel.assert_called_once_with(job_id="a" * 24, namespace="kofecheckmate")


if __name__ == "__main__":
    unittest.main()
