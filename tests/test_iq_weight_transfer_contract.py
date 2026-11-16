"""Offline regression tests for GLM-to-IQ transfer invariants."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[1] / "scripts" / "audit_iq_weight_transfer_contract.py"
spec = importlib.util.spec_from_file_location("audit_iq_weight_transfer_contract", script)
assert spec is not None and spec.loader is not None
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def fixture():
    donor = {
        "vocab_size": 154880, "hidden_size": 6144,
        "qk_head_dim": 256, "num_attention_heads": 64,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "num_hidden_layers": 78,
        "rms_norm_eps": 1e-5, "hidden_act": "silu",
    }
    target = {
        "model": {"vocab_size": 128256, "hidden_size": 4096,
                  "rms_norm_eps": 1e-5},
        "compressed_context": {"num_attention_heads": 64, "head_dim": 128},
        "moe": {"num_experts": 256, "top_k": 8, "expert_intermediate_size": 2048},
        "stable_moe": {"num_experts": 256, "top_k": 8,
                       "expert_intermediate_size": 2048},
        "moe_variant": "stable_latent",
        "attnres": {"block_size": 12},
        "schedule": {"layers": ["mamba3", "csa", "moe"] * 78 + ["mamba3"]},
    }
    manifest = {
        "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
        "source_layer_map": {str(i): i for i in range(78)},
    }
    return donor, target, manifest


class IQTransferInvariantContractTests(unittest.TestCase):
    def test_detects_all_four_categories(self) -> None:
        report = contract.audit(*fixture())
        self.assertEqual(report["status"], "blocked")
        bad = {f["category"] for f in report["findings"] if f["status"] != "pass"}
        self.assertEqual(bad, {"dimensions", "semantics", "mapping", "connectivity"})

    def test_preserved_expert_configuration_is_not_sufficient(self) -> None:
        report = contract.audit(*fixture())
        passing = {f["name"] for f in report["findings"] if f["status"] == "pass"}
        self.assertIn("routed_expert_count", passing)
        self.assertIn("routed_top_k", passing)
        self.assertIn("expert_intermediate_size", passing)
        self.assertEqual(report["status"], "blocked")

    def test_glm_shape_matching_cannot_fake_semantic_proof(self) -> None:
        donor, target, manifest = fixture()
        target["model"]["vocab_size"] = donor["vocab_size"]
        target["model"]["hidden_size"] = donor["hidden_size"]
        target["compressed_context"]["head_dim"] = donor["qk_head_dim"]
        report = contract.audit(donor, target, manifest)
        issues = {f["name"] for f in report["findings"] if f["status"] != "pass"}
        self.assertIn("expert_activation", issues)
        self.assertIn("tensor_layout_and_permutation_parity", issues)
        self.assertIn("physical_layer_count", issues)

    def test_duplicate_or_missing_layer_map_blocks(self) -> None:
        donor, target, manifest = fixture()
        manifest["source_layer_map"] = {"0": 0, "1": 0}
        report = contract.audit(donor, target, manifest)
        issues = {f["name"] for f in report["findings"] if f["status"] == "blocker"}
        self.assertIn("mapped_donor_layers", issues)
        self.assertIn("source_layer_indices", issues)


if __name__ == "__main__":
    unittest.main()
