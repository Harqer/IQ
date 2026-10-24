"""CPU-only isolated tests for exact early pre-MoE tap capture semantics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from iq_transfer.glm53_stream_capture import (
    GLM53StreamingCaptureError,
    capture_glm53_bootstrap_streaming,
)


class FakeConfig:
    def __init__(self, **kwargs):
        self.rms_norm_eps = 1e-5
        self._attn_implementation = "eager"


class FakeRotary:
    def __init__(self, config):
        self.config = config

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, hidden, position_ids=None):
        return None


class FakeSource:
    def __init__(self):
        self.normalized = []
        self.row_keys = []
        self.evicted = []

    def get_rows(self, key, row_ids):
        self.row_keys.append(key)
        # Use distinct values per token so capture cannot be all padding/zeros.
        return torch.arange(1, row_ids.numel() * 10 + 1, dtype=torch.float32).reshape(-1, 10)

    def clear_ephemeral_shards(self, **kwargs):
        self.evicted.append(kwargs)


class GLM53PreMoeEarlyCaptureTests(unittest.TestCase):
    def _capture(self, stage=3):
        config = {
            "model_type": "glm_moe_dsa",
            "hidden_size": 10,
            "num_hidden_layers": 78,
            "rms_norm_eps": 1e-5,
        }
        inspector = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=78, hidden_size=10),
            layout=SimpleNamespace(first_k_dense_replace=3, indexer_types=("full",) * 78),
        )
        x = FakeSource()
        attention_layers = []
        dense_calls = []

        class FakeAttention:
            def __call__(self, **kw):
                return torch.zeros_like(kw["hidden_states"]), None, None

        def load_attention(_source, *, layer, **kwargs):
            attention_layers.append(layer)
            if layer > 3:
                raise AssertionError("loaded attention beyond requested pre-MoE layer")
            return FakeAttention()

        def dense_mlp(_source, *, hidden, **kw):
            dense_calls.append(1)
            return torch.zeros_like(hidden), torch.zeros((1, 2, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with (
                patch("iq_transfer.glm53_stream_capture._require_runtime",
                      return_value=(torch, F, None, FakeConfig, object, FakeRotary)),
                patch("iq_transfer.glm53_stream_capture.GLM53Inspector.from_config_mapping",
                      return_value=inspector),
                patch("iq_transfer.glm53_stream_capture.canonical_glm53_config",
                      return_value=object()),
                patch("iq_transfer.glm53_stream_capture.bootstrap_glm53_source_layer_map",
                      return_value={0: 0, 1: 3}),
                patch("iq_transfer.glm53_stream_capture._causal_mask",
                      return_value=None),
                patch("iq_transfer.glm53_stream_capture._load_attention",
                      side_effect=load_attention),
                patch("iq_transfer.glm53_stream_capture._load_vector",
                      return_value=torch.ones(10)),
                patch("iq_transfer.glm53_stream_capture._dense_mlp",
                      side_effect=dense_mlp),
                patch("iq_transfer.glm53_stream_capture._sparse_moe",
                      side_effect=AssertionError("must not execute routed experts")),
            ):
                result = capture_glm53_bootstrap_streaming(
                    source=x,
                    config_path=path,
                    batches=[{
                        "input_ids": torch.tensor([[42, 43, 44]]),
                        "attention_mask": torch.tensor([[1, 1, 0]]),
                    }],
                    device="cpu",
                    stop_at_pre_moe_layer=stage,
                )
        return result, x, attention_layers, dense_calls

    def test_exact_pre_moe_input_capture_and_no_routed_download(self):
        result, source, layers, dense = self._capture()
        self.assertEqual(layers, [0, 1, 2, 3])
        self.assertEqual(len(dense), 3)
        self.assertEqual(source.row_keys, ["model.embed_tokens.weight"])
        self.assertEqual(set(result.spaces), {"layer.3.moe_input"})
        self.assertEqual(tuple(result.spaces["layer.3.moe_input"].shape), (2, 10))
        self.assertTrue(bool(torch.isfinite(result.spaces["layer.3.moe_input"]).all()))
        self.assertEqual(result.sample_count, 2)
        self.assertIn({"completed_layer": 3}, source.evicted)

    def test_rejects_dense_and_invalid_stop_layers_before_tensor_reads(self):
        for invalid in (0, 2, -1, 78, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(GLM53StreamingCaptureError):
                    self._capture(stage=invalid)


if __name__ == "__main__":
    unittest.main()
