"""Fail-fast contract for the exact full-width Colab runner.

This is a no-allocation META configuration test; do not misrepresent it
as having executed the production-size forward pass.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from iq_model import IQHybridForCausalLM
from iq_model.architecture import HybridLayerType
from iq_transfer.complete_transplant import canonical_complete_config


class ColabFullProductionContractTests(unittest.TestCase):
    def test_full_80_layer_4096_width_meta_construction(self):
        cfg = canonical_complete_config()
        self.assertEqual(cfg.model.hidden_size, 4096)
        self.assertEqual(cfg.model.vocab_size, 201088)
        self.assertEqual(
            [cfg.schedule.count(kind) for kind in (
                HybridLayerType.MAMBA3, HybridLayerType.CSA,
                HybridLayerType.HCA, HybridLayerType.MOE
            )],
            [32, 12, 12, 24]
        )
        self.assertEqual(cfg.mamba3.mimo_rank, 4)
        self.assertEqual(cfg.stable_moe.num_experts, 32)
        self.assertEqual(cfg.stable_moe.top_k, 4)
        self.assertEqual(cfg.stable_moe.latent_size, 2880)
        with torch.device("meta"):
            model = IQHybridForCausalLM(
                cfg, dtype=torch.float16, device="meta",
                test_only_cpu_reference=True
            )
        self.assertEqual(len(model.layers), 80)
        self.assertTrue(model.embed_tokens.weight.is_meta)
        self.assertTrue(model.lm_head.weight.is_meta)
        self.assertEqual(
            model.embed_tokens.weight.shape, (201088,4096))
        self.assertEqual(
            model.lm_head.weight.shape, (201088,4096))
        for i, kind in enumerate(cfg.schedule.layers):
            layer = model.layers[i]
            self.assertTrue(
                all(p.is_meta for p in layer.parameters()),
                f"Unintentionally materialized weight in layer {i}",
            )
            if kind is HybridLayerType.MOE:
                self.assertEqual(len(layer.moe.routed_experts),32)
            if kind in (HybridLayerType.CSA,HybridLayerType.HCA):
                self.assertEqual(
                    layer.attention.mode,
                    "csa" if kind is HybridLayerType.CSA else "hca",
                )

    def test_full_runner_parses_without_execution(self):
        script = Path(__file__).resolve().parents[1] / (
            "scripts/validate_full_iq_random_weights_colab.py"
        )
        parsed = ast.parse(script.read_text())
        self.assertTrue(
            any(isinstance(node,ast.FunctionDef) and
                node.name == "run_geometry" for node in parsed.body)
        )


if __name__ == "__main__":
    unittest.main()
