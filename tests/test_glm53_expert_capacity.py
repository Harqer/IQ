"""CPU-only expert-expansion compatibility checks (no model instantiated)."""
from __future__ import annotations

import unittest

from iq_transfer.complete_transplant import (
    CompleteTransplantError,
    canonical_glm53_config,
)


class TestGLM53ExpertCapacity(unittest.TestCase):
    def test_expert_variants_do_not_modify_mamba(self):
        base = canonical_glm53_config()
        self.assertEqual(base.stable_moe.num_experts, 32)
        self.assertEqual(base.moe.num_experts, 32)
        for count in (64, 128, 256):
            with self.subTest(count=count):
                more = canonical_glm53_config(routed_experts=count)
                self.assertEqual(more.stable_moe.num_experts, count)
                self.assertEqual(more.moe.num_experts, count)
                self.assertEqual(more.mamba3, base.mamba3)
                self.assertEqual(more.schedule, base.schedule)
                self.assertEqual(more.model, base.model)
                self.assertEqual(more.compressed_context, base.compressed_context)
                self.assertEqual(more.attnres, base.attnres)
                self.assertEqual(more.reasoning, base.reasoning)
                self.assertNotEqual(more.fingerprint, base.fingerprint)

    def test_unsupported_capacity_fails_closed(self):
        for value in (0, 31, 33, 257, True, 64.0):
            with self.subTest(value=value):
                with self.assertRaises(CompleteTransplantError):
                    canonical_glm53_config(routed_experts=value)


if __name__ == "__main__":
    unittest.main()
