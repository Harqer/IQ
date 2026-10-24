"""CPU fixture: exact 64-expert official-shard planning with no donor weights."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from iq_transfer.complete_transplant import canonical_glm53_config, donor_layer_positions
from iq_transfer.glm53 import GLM53Inspector
from iq_transfer.glm53_shards import _compile_tensor_keys

from scripts.plan_glm53_selective_shards import (
    SelectivePlanError, build_selective_plan, REVISION,
)


SOURCES = [
    0, 3, 6, 10, 14, 17, 22, 23, 26, 30, 34, 37, 42,
    44, 46, 50, 54, 57, 62, 64, 66, 70, 74, 77,
]


def fixture():
    config = {
        "model_type": "glm_moe_dsa",
        "dtype": "bfloat16",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "vocab_size": 154880,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "scoring_func": "sigmoid",
        "index_head_dim": 128,
        "index_n_heads": 32,
        "indexer_types": ["full"] * 78,
    }
    target = canonical_glm53_config(routed_experts=64)
    original = canonical_glm53_config()
    stages = []
    proxy = []
    data = {}
    integrity = {"donor_revision": REVISION, "files": {}}
    for i, ((ctx, moe), layer) in enumerate(zip(donor_layer_positions(original), SOURCES)):
        name = f"stage-{i:02d}-expert-usage.npy" if layer >= 3 else None
        usage = None
        if name is not None:
            usage = np.arange(256, dtype=np.float64) + 1
            byts = BytesIO()
            np.save(byts, usage)
            data[name] = byts.getvalue()
            integrity["files"][name] = sha256(data[name]).hexdigest()
        stages.append({
            "stage": i, "source_layer": layer,
            "context_physical_layer": ctx,
            "moe_physical_layer": moe,
            "expert_usage": name,
        })
        proxy.append(SimpleNamespace(
            stage=i, source_layer=layer,
            context_physical_layer=ctx, expert_usage=usage,
        ))
    inspector = GLM53Inspector.from_config_mapping(config)
    tensor_keys = _compile_tensor_keys(
        calibration=SimpleNamespace(stages=proxy),
        inspector=inspector,
        target_config=target,
    )
    # Model a genuine official index that contains additional donor tensors.
    index = {"weight_map": {
        key: f"model-{int(key.split('.')[2]) + 1:05d}-of-00282.safetensors"
        for key in tensor_keys
    }}
    index["weight_map"]["model.embed_tokens.weight"] = "model-00282-of-00282.safetensors"
    calibration = {
        "schema_version": 2,
        "calibration_mode": "weight_only",
        "source_layers": 78,
        "target_config_fingerprint": original.fingerprint,
        "stages": stages,
    }
    transplant = {
        "capability_source": {"repo": "zai-org/GLM-5.3-BF16", "revision": REVISION},
        "recipient_config_fingerprint": original.fingerprint,
        "source_layer_map": {str(i): layer for i, layer in enumerate(SOURCES)},
    }
    return config, index, calibration, integrity, data, transplant


class TestSelectiveGLMShardPlan(unittest.TestCase):
    def test_pinned_64_experts_requires_only_selected_layers(self):
        args = fixture()
        plan = build_selective_plan(*args, routed_experts=64)
        self.assertEqual(plan["status"], "selective_metadata_plan_only")
        self.assertEqual(plan["target_experts"], 64)
        self.assertEqual(len(plan["selected_donor_expert_ids"]), 24)
        self.assertIsNone(plan["selected_donor_expert_ids"][0])
        self.assertEqual(len(plan["selected_donor_expert_ids"][1]), 64)
        self.assertTrue(set(plan["official_shards"]) <= set(args[1]["weight_map"].values()))
        self.assertGreater(plan["tensor_keys_required"], 23 * 64 * 3)

    def test_reject_bad_stage_revision_and_expert_integrity(self):
        config, index, calibration, integrity, data, transplant = fixture()
        bad = dict(transplant)
        bad["capability_source"] = {**bad["capability_source"], "revision": "0" * 40}
        with self.assertRaisesRegex(SelectivePlanError, "donor revision"):
            build_selective_plan(config, index, calibration, integrity, data, bad, routed_experts=64)
        corrupt = dict(data)
        corrupt["stage-01-expert-usage.npy"] = b"tampered"
        with self.assertRaisesRegex(SelectivePlanError, "unverified"):
            build_selective_plan(config, index, calibration, integrity, corrupt, transplant, routed_experts=64)
        changed = dict(calibration)
        changed["target_config_fingerprint"] = "bad"
        with self.assertRaisesRegex(SelectivePlanError, "canonical"):
            build_selective_plan(config, index, changed, integrity, data, transplant, routed_experts=64)

    def test_unsupported_capacity_and_missing_tensors_fail(self):
        config, index, calibration, integrity, data, transplant = fixture()
        with self.assertRaises(SelectivePlanError):
            build_selective_plan(config, index, calibration, integrity, data, transplant, routed_experts=32)
        bad_index = {"weight_map": {"other": "missing.safetensors"}}
        with self.assertRaises(Exception):
            build_selective_plan(config, bad_index, calibration, integrity, data, transplant, routed_experts=64)


if __name__ == "__main__":
    unittest.main()
