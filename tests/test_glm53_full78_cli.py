"""CPU-only CLI safety tests: never submit an H200 or contact Hugging Face."""
from __future__ import annotations

import io
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
