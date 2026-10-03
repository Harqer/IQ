from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iq_transfer.glm53 import GLM53Inspector
from iq_transfer.glm53_job import (
    GLM53TransferError,
    validate_glm53_streaming_donor,
)


def config() -> dict[str, object]:
    return {
        "_name_or_path": "zai-org/GLM-5.3-BF16",
        "model_type": "glm_moe_dsa",
        "dtype": "bfloat16",
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 3,
        "qk_rope_head_dim": 1,
        "v_head_dim": 4,
        "moe_intermediate_size": 6,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 1,
        "scoring_func": "sigmoid",
        "index_head_dim": 4,
        "index_n_heads": 2,
        "indexer_types": ["full", "shared"],
    }


def write_metadata(root: Path, *, drop_key: str | None = None) -> None:
    cfg = config()
    inspector = GLM53Inspector.from_config_mapping(cfg)
    keys = list(inspector.required_tensor_keys())
    if drop_key is not None:
        keys.remove(drop_key)
    (root / "config.json").write_text(
        json.dumps(cfg),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    key: "model-00001-of-00001.safetensors"
                    for key in keys
                },
            }
        ),
        encoding="utf-8",
    )
class GLM53StreamingValidationTests(unittest.TestCase):
    def test_streaming_manifest_uses_only_metadata_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metadata(root)
            artifact = validate_glm53_streaming_donor(
                checkpoint=root,
                checkpoint_revision="abc123",
                donor_license="GLM-5.3",
                source_uri="hf://models/zai-org/GLM-5.3-BF16@abc123",
            )
        self.assertEqual(artifact.manifest.schema_version, 2)
        self.assertEqual(
            {item.name for item in artifact.manifest.files},
            {"config.json", "model.safetensors.index.json"},
        )
        self.assertEqual(artifact.manifest.tensors, ())
        self.assertIn(
            "streaming-index",
            artifact.manifest.operator_layout_version,
        )

    def test_streaming_validation_rejects_incomplete_index(self):
        missing = "model.layers.1.mlp.experts.1.down_proj.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metadata(root, drop_key=missing)
            with self.assertRaises(GLM53TransferError):
                validate_glm53_streaming_donor(
                    checkpoint=root,
                    checkpoint_revision="abc123",
                    donor_license="GLM-5.3",
                )


if __name__ == "__main__":
    unittest.main()
