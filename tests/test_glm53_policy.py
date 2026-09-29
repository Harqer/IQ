from __future__ import annotations

import unittest

from iq_transfer.glm53_policy import (
    GLM53TransferDisposition,
    IQ_RECIPIENT_NATIVE_FAMILIES,
    classify_glm53_source_role,
)


class GLM53TransferPolicyTests(unittest.TestCase):
    def test_lexical_weights_use_coordinate_transport(self):
        self.assertEqual(
            classify_glm53_source_role("embedding").disposition,
            GLM53TransferDisposition.OPERATOR_TRANSPORT,
        )
        self.assertEqual(
            classify_glm53_source_role("lm_head").disposition,
            GLM53TransferDisposition.OPERATOR_TRANSPORT,
        )

    def test_mla_dsa_and_moe_require_functional_transfer(self):
        for role in (
            "attn.q_a",
            "attn.kv_b",
            "dsa.indexer.q",
            "mlp.gate",
            "moe.router",
            "moe.shared.down",
            "moe.expert.42.up",
        ):
            with self.subTest(role=role):
                self.assertEqual(
                    classify_glm53_source_role(role).disposition,
                    GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
                )

    def test_norms_and_iq_only_state_are_recipient_native(self):
        self.assertEqual(
            classify_glm53_source_role("norm.input").disposition,
            GLM53TransferDisposition.RECIPIENT_NATIVE,
        )
        self.assertIn("mamba3.recurrence", IQ_RECIPIENT_NATIVE_FAMILIES)
        self.assertIn("block_attnres", IQ_RECIPIENT_NATIVE_FAMILIES)

    def test_unknown_source_role_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unclassified"):
            classify_glm53_source_role("unknown.magic")


if __name__ == "__main__":
    unittest.main()
