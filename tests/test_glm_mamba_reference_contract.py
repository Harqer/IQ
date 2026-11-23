"""Zero-compute regression tests for official source reference integrity."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
file = HERE / "scripts" / "audit_glm_mamba_references.py"
spec = importlib.util.spec_from_file_location("audit_glm_mamba_references", file)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference_configs():
    glm = {
        "model_type": "glm_moe_dsa", "vocab_size": 154880,
        "hidden_size": 6144, "num_hidden_layers": 78, "num_attention_heads": 64,
        "q_lora_rank": 2048, "kv_lora_rank": 512,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "qk_head_dim": 256,
        "v_head_dim": 256, "moe_intermediate_size": 2048,
        "intermediate_size": 12288, "first_k_dense_replace": 3,
        "n_routed_experts": 256, "n_shared_experts": 1,
        "num_experts_per_tok": 8, "hidden_act": "silu",
        "scoring_func": "sigmoid", "norm_topk_prob": True,
        "routed_scaling_factor": 2.5, "rms_norm_eps": 1e-5,
        "rope_interleave": True, "tie_word_embeddings": False,
        "eos_token_id": [154820, 154827, 154829],
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
    }
    mamba = {
        "vocab_size": 128256, "d_model": 2048, "n_layer": 24,
        "d_intermediate": 3824, "attn_layer_idx": [],
        "tie_embeddings": True,
        "ssm_cfg": {"layer": "Mamba3", "d_state": 128,
                    "expand": 2, "headdim": 64, "is_mimo": True,
                    "mimo_rank": 4, "chunk_size": 16, "rope_fraction": 0.5},
    }
    return glm, mamba


class ReferenceContractTests(unittest.TestCase):
    def test_official_configs_pass_source_only_gate(self):
        result = module.audit_sources(*reference_configs())
        self.assertEqual(result["status"], "reference_config_consistent")
        self.assertEqual(result["checks"]["mamba_complete_blocks"], 24)

    def test_mimo_rank_is_not_mlp_cadence(self):
        glm, mamba = reference_configs()
        mamba["d_intermediate"] = 0
        report = module.audit_sources(glm, mamba)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("mamba.d_intermediate",
                      {x["key"] for x in report["findings"]})

    def test_wrong_glm_vocab_and_special_tokens_rejected(self):
        glm, mamba = reference_configs()
        glm["vocab_size"] = mamba["vocab_size"]
        glm["eos_token_id"] = [128001, 128009]
        report = module.audit_sources(glm, mamba)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("glm.eos_token_ids",
                      {x["key"] for x in report["findings"]})

    def test_mamba_lexical_checkpoint_is_not_glm_first(self):
        glm, mamba = reference_configs()
        manifest = {
            "foundation": {"repo": module.MAMBA_SOURCE,
                           "revision": module.PINNED_MAMBA_REVISION},
            "capability_source": {"repo": module.GLM_SOURCE,
                                  "revision": module.PINNED_GLM_REVISION},
            "calibration_mode": "weight_only",
        }
        status = {"inference_ready": False,
                  "end_to_end_knowledge_retention_proven": False}
        r = module.audit_sources(glm, mamba, manifest=manifest,
                                 compile_status=status, lexical_intent="glm53")
        self.assertEqual(r["status"], "blocked")
        keys = {f["key"] for f in r["findings"]}
        self.assertIn("checkpoint.lexical_owner", keys)
        self.assertIn("checkpoint.inference_ready", keys)

    def test_accidental_siso_conversion_fails(self):
        glm, mamba = reference_configs()
        mamba["ssm_cfg"]["is_mimo"] = False
        result = module.audit_sources(glm, mamba)
        self.assertIn("mamba.ssm.is_mimo", {f["key"] for f in result["findings"]})


if __name__ == "__main__":
    unittest.main()
