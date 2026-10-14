"""Fast CPU contract tests for GLM5Bridge donor index (no fabricated weights)."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from iq_transfer.glm53_donor_preflight import (
    GLM53DonorPreflightError, audit_glm53_donor,
    expected_glm53_tensor_shape, required_glm53_hf_keys, _validate_shard,
)
from iq_transfer.glm53_expert_mapping import (
    iter_glm53_expert_transfer_tasks,
    glm53_expert_transfer_manifest,
)
from iq_transfer.glm53_parallel_contract import (
    GLM53Topology, expected_indexer_types,
)


def published_glm53_config() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "indexer_types": list(expected_indexer_types()),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "num_nextn_predict_layers": 1,
        "vocab_size": 154880,
    }


def write_index_only(root: Path, keys: tuple[str, ...]) -> None:
    (root / "config.json").write_text(
        json.dumps(published_glm53_config()), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "donor.safetensors" for name in keys}}),
        encoding="utf-8"
    )
    # This is *index-only* validation; no model tensor data is fabricated.
    (root / "donor.safetensors").touch()


class OfficialDonorPreflightContractTests(unittest.TestCase):
    def test_complete_78_stages_require_all_256_expert_ids_per_sparse_stage(self):
        topology = GLM53Topology.from_hf_config(published_glm53_config())
        keys = required_glm53_hf_keys(topology)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            len([k for k in keys if ".mlp.experts." in k]), 75 * 256 * 3
        )
        for block in (3, 27, 77):
            for expert in (0, 128, 255):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    self.assertIn(
                        f"model.layers.{block}.mlp.experts.{expert}.{projection}.weight",
                        keys,
                    )
        self.assertNotIn(
            "model.layers.78.mlp.experts.0.gate_proj.weight", keys
        )
        self.assertIn("model.layers.0.mlp.gate_proj.weight", keys)
        self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", keys)
        self.assertIn(
            "model.layers.2.self_attn.indexer.wq_b.weight", keys
        )
        self.assertNotIn(
            "model.layers.3.self_attn.indexer.wq_b.weight", keys
        )

    def test_real_index_missing_single_expert_fails_before_conversion(self):
        topology = GLM53Topology.from_hf_config(published_glm53_config())
        keys = required_glm53_hf_keys(topology)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_index_only(root, keys)
            report = audit_glm53_donor(
                root, require_bf16=False, verify_headers=False,
                checkpoint_identity="zai-org/GLM-5.3",
            )
            self.assertFalse(report.complete)
            self.assertFalse(report.verified_headers)
            self.assertEqual(report.required_keys, len(keys))
            self.assertEqual(report.layer_counts[77] > 768, True)
            self.assertEqual(report.referenced_shards, 1)
            self.assertEqual(len(report.config_sha256), 64)
            self.assertEqual(len(report.index_sha256), 64)
            self.assertFalse(report.to_dict()["converted_weights"])

            missing = "model.layers.77.mlp.experts.255.down_proj.weight"
            write_index_only(root, tuple(k for k in keys if k != missing))
            with self.assertRaisesRegex(GLM53DonorPreflightError, "missing 1"):
                audit_glm53_donor(
                    root, require_bf16=False, verify_headers=False,
                    checkpoint_identity="zai-org/GLM-5.3",
                )

    def test_native_bf16_requires_actual_tensor_header_verification(self):
        topology = GLM53Topology.from_hf_config(published_glm53_config())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_index_only(root, required_glm53_hf_keys(topology))
            with self.assertRaisesRegex(GLM53DonorPreflightError, "requires"):
                audit_glm53_donor(
                    root, require_bf16=True, verify_headers=False,
                    checkpoint_identity="zai-org/GLM-5.3-BF16",
                )
            with self.assertRaisesRegex(GLM53DonorPreflightError, "publisher"):
                audit_glm53_donor(
                    root, require_bf16=False, verify_headers=False,
                    checkpoint_identity="untrusted/GLM-5.3",
                )

    def test_strict_dense_and_sparse_shapes_from_official_glm_bridge(self):
        self.assertEqual(
            expected_glm53_tensor_shape(
                "model.layers.0.mlp.gate_proj.weight", vocab_size=154880
            ), (12288, 6144)
        )
        self.assertEqual(
            expected_glm53_tensor_shape(
                "model.layers.77.mlp.experts.255.down_proj.weight",
                vocab_size=154880,
            ), (6144, 2048)
        )
        self.assertEqual(
            expected_glm53_tensor_shape(
                "model.layers.18.mlp.shared_experts.up_proj.weight",
                vocab_size=154880,
            ), (2048, 6144)
        )
        self.assertEqual(
            expected_glm53_tensor_shape(
                "model.layers.50.mlp.gate.weight", vocab_size=154880
            ), (256, 6144)
        )
        self.assertEqual(
            expected_glm53_tensor_shape("lm_head.weight", vocab_size=154880),
            (154880, 6144),
        )

    def test_rank_owned_source_to_destination_registry_covers_every_expert(self):
        tasks = list(iter_glm53_expert_transfer_tasks(8))
        self.assertEqual(
            sum(task.action == "routed_expert_latent_transport" for task in tasks),
            75 * 256 * 3,
        )
        self.assertEqual(len(tasks), 3 * 3 + 75 * (5 + 256 * 3))
        self.assertEqual(len({t.source for t in tasks}), len(tasks))
        self.assertEqual(
            len({(t.owner, t.destination) for t in tasks}), len(tasks)
        )
        for rank in range(8):
            owned = [t for t in tasks if t.owner == rank and
                     t.action == "routed_expert_latent_transport"]
            self.assertEqual(len(owned), 75 * 32 * 3)
        first = next(t for t in tasks if t.source ==
                     "model.layers.3.mlp.experts.32.gate_proj.weight")
        self.assertEqual(first.destination,
                         "layers.3.moe.moe.routed_experts.0.gate_proj.weight")
        self.assertEqual(first.owner, 1)
        manifest = glm53_expert_transfer_manifest(8)
        self.assertFalse(manifest["conversion_executed"])
        self.assertFalse(manifest["attention_mapping_complete"])
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertEqual(manifest, glm53_expert_transfer_manifest(8))

    def test_safetensors_index_paths_do_not_escape_source_root(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "valid.safetensors").touch()
            self.assertEqual(
                _validate_shard(root, "valid.safetensors"),
                root / "valid.safetensors",
            )
            for bad in ("../secret.safetensors", "nested/donor.safetensors",
                        "/etc/passwd", "dir\\donor.safetensors"):
                with self.assertRaises(GLM53DonorPreflightError):
                    _validate_shard(root, bad)


if __name__ == "__main__":
    unittest.main()
