from __future__ import annotations

import json
import unittest

from iq_transfer.nvidia_foundation import (
    NVIDIA_BASE_REPO,
    NvidiaFoundationError,
    NvidiaDonorGeometry,
    plan_foundation,
)

# Published NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 config geometry.
# The production preflight fetches and pins the actual config from Hugging Face.
OFFICIAL_GEOMETRY = {
    "model_type": "nemotron_h",
    "vocab_size": 131072,
    "hidden_size": 4096,
    "num_hidden_layers": 88,
    "hybrid_override_pattern": "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
    "moe_latent_size": 1024,
    "n_routed_experts": 512,
    "num_experts_per_tok": 22,
}


class NvidiaFoundationPreflightTests(unittest.TestCase):
    def test_official_config_geometry(self):
        result = NvidiaDonorGeometry.from_config(OFFICIAL_GEOMETRY)
        self.assertEqual(result.layer_count, 88)
        self.assertEqual(
            (result.mamba2_layers, result.attention_layers, result.moe_layers),
            (40, 8, 40),
        )

    def test_preflight_records_exact_donor_sha_and_separate_operator_roles(self):
        config_bytes = json.dumps(OFFICIAL_GEOMETRY).encode()
        plan = plan_foundation(
            OFFICIAL_GEOMETRY,
            donor_revision="a" * 40,
            config_bytes=config_bytes,
        )
        self.assertEqual(plan.donor_repo, NVIDIA_BASE_REPO)
        self.assertEqual(plan.recipient_parallel_blocks, 78)
        self.assertEqual(plan.recipient_hidden_size, 4096)
        self.assertFalse(plan.direct_mamba2_to_mamba3)
        self.assertFalse(plan.same_tokenizer_as_mamba3)
        self.assertEqual(len(plan.donor_config_sha256), 64)

    def test_cannot_claim_mamba2_is_mamba3(self):
        incorrect = dict(OFFICIAL_GEOMETRY, model_type="falcon_h1")
        with self.assertRaisesRegex(NvidiaFoundationError, "nemotron_h"):
            NvidiaDonorGeometry.from_config(incorrect)

    def test_fails_closed_on_depth_and_pattern_mismatch(self):
        incorrect = dict(OFFICIAL_GEOMETRY, num_hidden_layers=78)
        with self.assertRaisesRegex(NvidiaFoundationError, "pattern"):
            NvidiaDonorGeometry.from_config(incorrect)

    def test_fails_closed_on_wrong_donor_width(self):
        incorrect = dict(OFFICIAL_GEOMETRY, hidden_size=2688)
        with self.assertRaisesRegex(NvidiaFoundationError, "Unexpected NVIDIA"):
            NvidiaDonorGeometry.from_config(incorrect)

    def test_rejects_unpinned_revision(self):
        with self.assertRaisesRegex(NvidiaFoundationError, "commit SHA"):
            plan_foundation(
                OFFICIAL_GEOMETRY,
                donor_revision="main",
                config_bytes=b"real checkpoint config",
            )


if __name__ == "__main__":
    unittest.main()
