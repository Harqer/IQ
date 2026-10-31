"""Read-only storage policy tests: no HF credentials or actual bucket writes."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_glm53_storage_policy import (
    DEFAULT_BUCKET, StoragePolicyError, verify_bucket_policy,
)


class PublicAndPrivateOutputTests(unittest.TestCase):
    @patch("scripts.check_glm53_storage_policy.bucket_info",
           return_value=SimpleNamespace(private=False))
    def test_public_bucket_requires_explicit_best_effort_ack(self, _):
        with self.assertRaisesRegex(StoragePolicyError, "best-effort"):
            verify_bucket_policy(DEFAULT_BUCKET)
        accepted = verify_bucket_policy(
            DEFAULT_BUCKET, allow_public_best_effort=True,
        )
        self.assertEqual(accepted["visibility"], "public")
        self.assertEqual(accepted["policy"], "best_effort_not_guaranteed")
        self.assertTrue(any("BILLABLE" in warning
                            for warning in accepted["warnings"]))

    @patch("scripts.check_glm53_storage_policy.bucket_info",
           return_value=SimpleNamespace(private=True))
    def test_private_bucket_never_accepts_public_override(self, _):
        with self.assertRaisesRegex(StoragePolicyError, "provider-verified"):
            verify_bucket_policy(
                DEFAULT_BUCKET, allow_public_best_effort=True,
            )

    @patch("scripts.check_glm53_storage_policy.check_capacity",
           return_value={"free_bytes": 700_000_000_000})
    @patch("scripts.check_glm53_storage_policy.bucket_info",
           return_value=SimpleNamespace(private=True))
    def test_private_capacity_requires_exact_verified_attestation(
        self, _, verifier,
    ):
        report = verify_bucket_policy(
            DEFAULT_BUCKET, private_attestation="/evidence/private.json"
        )
        self.assertEqual(report["visibility"], "private")
        self.assertEqual(report["verified_free_bytes"], 700_000_000_000)
        self.assertEqual(verifier.call_args.args[-1], DEFAULT_BUCKET)

    @patch("scripts.check_glm53_storage_policy.bucket_info")
    def test_never_assume_unsupported_bucket_or_provider_state(self, provider):
        with self.assertRaisesRegex(StoragePolicyError, "unexpected"):
            verify_bucket_policy("someone/other-bucket", allow_public_best_effort=True)
        provider.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(StoragePolicyError, "cannot verify"):
            verify_bucket_policy(DEFAULT_BUCKET, allow_public_best_effort=True)


if __name__ == "__main__":
    unittest.main()
