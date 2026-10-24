"""Check official router selection and strict capture provenance without downloading weights."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iq_transfer.glm53_routing_replay import (
    GLM53RoutingReplayError,
    read_verified_capture,
    select_routed_experts,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class RouterReplayTests(unittest.TestCase):
    def test_router_uses_correction_bias_and_deterministic_topk(self):
        hidden = np.array([
            [2., 0., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float32)
        weight = torch.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        bias = torch.tensor([0., -0.1, 0., 0.4])
        config = {
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
        }
        selected = select_routed_experts(hidden, weight, bias, config)
        self.assertEqual(selected.shape, (2, 2))
        self.assertEqual(set(selected[0].tolist()), {0, 3})
        self.assertEqual(set(selected[1].tolist()), {2, 3})

    def test_rejects_router_geometry(self):
        values = np.ones((2, 4), dtype=np.float32)
        weights = torch.ones(4, 4)
        config = {
            "n_routed_experts": 4, "num_experts_per_tok": 2,
            "n_group": 1, "topk_group": 1,
            "routed_scaling_factor": 2.5,
        }
        with self.assertRaises(GLM53RoutingReplayError):
            select_routed_experts(values, weights, torch.ones(3), config)

    def test_capture_sha_and_metadata_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "tokens.npy"
            values = np.ones((2, 6144), dtype=np.float32)
            np.save(capture, values)
            digest = hashlib.sha256(capture.read_bytes()).hexdigest()
            metadata = {
                "artifact_type": "glm53_real_pre_moe_hidden_capture",
                "model_repo": REPO,
                "model_revision": REVISION,
                "donor_layer": 3,
                "tap": "layer.3.moe_input",
                "tap_position": "after post_attention_layernorm, before sparse MoE",
                "official_config_sha256": "config-sha",
                "official_index_sha256": "index-sha",
                "hidden_width": 6144,
                "dtype": "float32",
                "num_valid_tokens": 2,
                "sha256": digest,
            }
            capture.with_suffix(".json").write_text(json.dumps(metadata))
            verified = read_verified_capture(
                capture, config_sha256="config-sha", index_sha256="index-sha"
            )
            self.assertEqual(verified.shape, (2, 6144))
            values[0, 0] = 0
            np.save(capture, values)
            with self.assertRaisesRegex(
                GLM53RoutingReplayError, "checksum mismatch"
            ):
                read_verified_capture(
                    capture, config_sha256="config-sha", index_sha256="index-sha"
                )


if __name__ == "__main__":
    unittest.main()
