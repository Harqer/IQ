"""Deterministic IQ 78-block destination coverage contract tests."""
from __future__ import annotations

from dataclasses import replace
import unittest
import torch

from iq_transfer.parallel_parameter_plan import (
    ParameterRecord, ParallelMappingError,
    plan_global_inventory, collect_local_inventory,
)
from iq_transfer.complete_transplant import nvidia_parallel_78_config
from iq_model.config import IQModelConfig
from iq_model.mlp.swiglu import SwiGLU
from iq_model.norm import RMSNorm


def names_for_topology() -> list[ParameterRecord]:
    rows = []
    for layer in range(78):
        family = "dense_ffn" if layer < 3 else "moe"
        for token in ("norm.weight", "mamba.core.out_proj.weight",
                      "attention.q_a_proj.weight", "fusion.projection.weight",
                      "expert_gate.scale", f"{family}.weight"):
            # Names and sizes here test the metadata algorithm; actual weights
            # must be resolved from named_parameters on the distributed worker.
            rows.append(ParameterRecord(
                global_name=f"layers.{layer}.{token}",
                shape=(4096,), dtype="torch.bfloat16",
                kind="parameter", pp_rank=0, tp_rank=0, ep_rank=0,
            ))
    rows.append(ParameterRecord(
        "embed_tokens.weight", (131072, 4096),
        "torch.bfloat16", "parameter", 0, 0, 0
    ))
    return rows


class MappingPlanTests(unittest.TestCase):
    def test_deterministic_full_78_block_coverage(self):
        config = nvidia_parallel_78_config()
        original = names_for_topology()
        a = plan_global_inventory(original, config)
        b = plan_global_inventory(list(reversed(original)), config)
        self.assertEqual(a, b)
        self.assertEqual(len(a["tasks"]), 78 * 6 + 1)
        self.assertEqual(a["status"], "preflight_only_not_weight_converted")
        self.assertEqual(a["source_counts"]["glm53"], 78 + 78)
        self.assertTrue(all(not task["verified"] for task in a["tasks"]))
        self.assertEqual(len(a["sha256"]), 64)

    def test_missing_expert_stage_and_duplicate_rank_local_key_are_rejected(self):
        config = nvidia_parallel_78_config()
        original = names_for_topology()
        without_expert = [p for p in original if p.global_name != "layers.4.moe.weight"]
        with self.assertRaisesRegex(ParallelMappingError, "missing parameter families"):
            plan_global_inventory(without_expert, config)
        with self.assertRaisesRegex(ParallelMappingError, "duplicate rank-local"):
            plan_global_inventory(original + [original[-1]], config)

    def test_local_parameter_collection_uses_model_names_and_global_pp_indices(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

        real = Tiny()
        rows = collect_local_inventory(
            real, pp_rank=1, local_to_global={0: 39}
        )
        self.assertEqual(rows[0].global_name, "layers.39.weight")
        self.assertEqual(rows[0].shape, (2, 2))
        with self.assertRaisesRegex(ParallelMappingError, "global PP index"):
            collect_local_inventory(real, pp_rank=1)

    def test_real_dense_swiglu_has_three_distinct_projections_and_gradient(self):
        base = nvidia_parallel_78_config().model
        small = replace(base, vocab_size=64, hidden_size=16,
                        intermediate_size=48, num_attention_heads=4,
                        num_key_value_heads=2)
        ffn = SwiGLU(small)
        norm = RMSNorm(16, small.rms_norm_eps)
        x = torch.randn(2, 5, 16, requires_grad=True)
        y = ffn(norm(x))
        self.assertEqual(tuple(y.shape), (2, 5, 16))
        y.square().mean().backward()
        self.assertIsNotNone(ffn.gate_proj.weight.grad)
        self.assertIsNotNone(ffn.up_proj.weight.grad)
        self.assertIsNotNone(ffn.down_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
