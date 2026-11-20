"""CPU-only metadata gate, malicious paths and non-destructive audit."""
from __future__ import annotations
from copy import deepcopy
import json
from hashlib import sha256
from pathlib import Path
import unittest
from iq_cloud.checkpoint_metadata import (
    inspect_index, check_pinned_metadata_sha256, IQ235MetadataError,
    EXPECTED_CONFIG_SHA256, EXPECTED_INDEX_SHA256,
)
root = Path(__file__).resolve().parents[1]
source = json.loads((root / "configs/iq235_checkpoint_source.json").read_text())
target = json.loads((root / "configs/iq235_glm53_tokenizer_phase_a.json").read_text())
marker = dict(status="FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED",
              inference_ready=False, tokenizer_packaged=False,
              end_to_end_knowledge_retention_proven=False,
              mapped_layers=78, mapped_routed_expert_slots=19200,
              routed_experts_per_layer=256)
idx = {"metadata": {"total_size": 100}, "weight_map": {
    "embed_tokens.weight": "model-global.safetensors",
    "lm_head.weight": "model-global.safetensors",
    "norm.weight": "model-global.safetensors",
    "layers.0.norm.weight": "model-mamba-000.safetensors",
}}
class MetadataTests(unittest.TestCase):
    def gate(self, index=None, status=None):
        return inspect_index(source, source, target,
                             idx if index is None else index,
                             marker if status is None else status,
                             counts=(4, 2, 100))
    def test_source_metadata_is_compatible_not_inference_ready(self):
        result = self.gate()
        self.assertEqual(result["model_layers"], 235)
        self.assertFalse(result["inference_ready"])
        self.assertFalse(result["weight_payloads_read"])
    def test_bad_shard_path_blocked(self):
        corrupt = deepcopy(idx)
        corrupt["weight_map"]["layers.0.norm.weight"] = "../bad.safetensors"
        with self.assertRaises(IQ235MetadataError):
            self.gate(index=corrupt)
    def test_missing_global_lexical_tensor_blocked(self):
        corrupt = deepcopy(idx)
        corrupt["weight_map"]["lm_head.weight"] = "model-mamba-000.safetensors"
        with self.assertRaises(IQ235MetadataError):
            self.gate(index=corrupt)
    def test_completion_status_cannot_claim_ready(self):
        bad = dict(marker, inference_ready=True)
        with self.assertRaises(IQ235MetadataError):
            self.gate(status=bad)
    def test_bad_total_bytes_blocked(self):
        corrupt = deepcopy(idx)
        corrupt["metadata"]["total_size"] += 1
        with self.assertRaises(IQ235MetadataError):
            self.gate(index=corrupt)
    def test_pinned_hashes_fail_closed_on_config_and_index_changes(self):
        config = b'{"model":{"vocab_size":128256}}'
        index = b'{"weight_map":{"lm_head.weight":"model-global.safetensors"}}'
        good_cfg = sha256(config).hexdigest()
        good_idx = sha256(index).hexdigest()
        check_pinned_metadata_sha256(
            config, index, config_digest=good_cfg, index_digest=good_idx
        )
        with self.assertRaisesRegex(IQ235MetadataError, "config SHA-256"):
            check_pinned_metadata_sha256(
                config + b" ", index, config_digest=good_cfg, index_digest=good_idx
            )
        with self.assertRaisesRegex(IQ235MetadataError, "index SHA-256"):
            check_pinned_metadata_sha256(
                config, index + b" ", config_digest=good_cfg, index_digest=good_idx
            )
        self.assertEqual(len(EXPECTED_CONFIG_SHA256), 64)
        self.assertEqual(len(EXPECTED_INDEX_SHA256), 64)

    def test_preflight_does_not_call_staging_mutator(self):
        text = (root / "scripts/iq235_preflight.py").read_text()
        self.assertNotIn("stage_sources", text)
        self.assertIn("selected_sources()", text)
        self.assertIn('"staging_performed": False', text)

if __name__ == "__main__":
    unittest.main()
