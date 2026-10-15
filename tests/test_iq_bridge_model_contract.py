"""NVIDIA-documented pipeline layout and IQ HF config identity acceptance.

No fake Megatron model or checkpoint weights. These tests verify the actual
78 block geometry and that impossible cross-stage IndexShare is rejected.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from transformers import AutoConfig

from iq_training.pipeline_contract import (
    IQPipelineContractError, partition_iq_78,
)
from iq_training.hf_model_config import (
    IQHFCausalConfig, IQHFConfigError, register_iq_hf_config,
)
from iq_transfer.complete_transplant import nvidia_parallel_78_config


class IQHuggingFaceIdentityTests(unittest.TestCase):
    def test_real_transformers_config_round_trip_and_registered_auto_config(self):
        config = IQHFCausalConfig()
        backbone = nvidia_parallel_78_config()
        self.assertEqual(config.model_type, "iq_mamba3_csa")
        self.assertEqual(config.iq_hybrid_sha256, backbone.fingerprint)
        self.assertEqual(config.vocab_size, 131072)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 78)
        self.assertEqual(len(config.validated_iq_hybrid().schedule.moe_positions), 75)
        with TemporaryDirectory() as temp:
            config.save_pretrained(temp)
            path = Path(temp) / "config.json"
            data = json.loads(path.read_text())
            self.assertEqual(data["model_type"], config.model_type)
            self.assertEqual(data["iq_hybrid_sha256"], backbone.fingerprint)
            register_iq_hf_config()
            restored = AutoConfig.from_pretrained(temp, local_files_only=True)
            self.assertIsInstance(restored, IQHFCausalConfig)
            self.assertEqual(restored.validated_iq_hybrid().fingerprint, backbone.fingerprint)
            self.assertEqual(restored.to_dict()["iq_hybrid"], config.to_dict()["iq_hybrid"])

    def test_rejects_forged_shape_and_fingerprint(self):
        model = nvidia_parallel_78_config()
        with self.assertRaisesRegex(IQHFConfigError, "fingerprint mismatch"):
            IQHFCausalConfig(iq_hybrid=model.to_dict(), iq_hybrid_sha256="bad")
        with self.assertRaisesRegex(IQHFConfigError, "contradicts"):
            IQHFCausalConfig(iq_hybrid=model.to_dict(), hidden_size=6144)
        cfg = IQHFCausalConfig()
        cfg.iq_hybrid["model"]["vocab_size"] = 300
        with self.assertRaises(IQHFConfigError):
            cfg.validated_iq_hybrid()


class IQMegatronPipelineLayoutTests(unittest.TestCase):
    def test_complete_78_layers_single_rank_and_full_model_blockers(self):
        cfg = nvidia_parallel_78_config(expert_parallel_size=8)
        plan = partition_iq_78(cfg)
        self.assertEqual(plan.pp_size, 1)
        stage = plan.stages[0]
        self.assertEqual(len(stage.layers), 78)
        self.assertEqual(stage.dense_blocks, (0, 1, 2))
        self.assertEqual(stage.moe_blocks, tuple(range(3, 78)))
        self.assertEqual(len(stage.index_owner_blocks), 21)
        self.assertEqual(plan.megatron_layer_layout()[0][0], "embedding")
        self.assertEqual(plan.megatron_layer_layout()[0][-1], "loss")
        self.assertEqual(plan.megatron_layer_layout()[0].count("decoder"), 78)
        self.assertEqual(plan, partition_iq_78(cfg))
        manifest = plan.to_dict()
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertFalse(manifest["executable_as_megatron_bridge_provider"])
        self.assertTrue(manifest["blocked_capabilities"])
        with self.assertRaisesRegex(IQPipelineContractError, "not registered"):
            plan.require_mcore_training_ready()

    def test_disallow_indexshare_cross_rank_and_dependent_attnres(self):
        cfg = nvidia_parallel_78_config()
        # Full source indexer layer 2 must remain with shared layers 3..5.
        for cuts in ((0, 3, 78), (0, 4, 78), (0, 5, 78)):
            with self.assertRaisesRegex(IQPipelineContractError, "IndexShare"):
                partition_iq_78(cfg, boundaries=cuts)
        # IndexShare owner 6 starts a new source group, so this cut is
        # legal as a *logical* partition. Execution still requires cross
        # stage Block AttnRes handling.
        plan = partition_iq_78(cfg, boundaries=(0, 6, 10, 78))
        self.assertEqual(
            [(s.start_layer, s.end_layer) for s in plan.stages],
            [(0, 6), (6, 10), (10, 78)],
        )
        self.assertEqual(
            [len(stage.moe_blocks) for stage in plan.stages], [3, 4, 68]
        )
        self.assertTrue(any("AttnRes" in issue for issue in plan.blocked_capabilities))
        self.assertEqual(sum(row.count("decoder") for row in plan.megatron_layer_layout()), 78)

    def test_malformed_cuts_and_altered_architecture_rejected(self):
        cfg = nvidia_parallel_78_config()
        for edges in ((0, 78, 78), (1, 78), (0, 80), (0, 6, 5, 78)):
            with self.assertRaisesRegex(IQPipelineContractError, "boundaries"):
                partition_iq_78(cfg, boundaries=edges)
        changed = replace(cfg, stable_moe=replace(cfg.stable_moe, num_experts=32))
        with self.assertRaisesRegex(IQPipelineContractError, "78-block"):
            partition_iq_78(changed)

    def test_actual_megatron_core_layout_parser_if_installed(self):
        try:
            from megatron.core.transformer.pipeline_parallel_layer_layout import (
                PipelineParallelLayerLayout,
            )
        except ImportError:
            self.skipTest("MCore is optional for standard CPU model test job")
        plan = partition_iq_78(
            nvidia_parallel_78_config(), boundaries=(0, 6, 10, 78),
        )
        parsed = PipelineParallelLayerLayout(
            plan.megatron_layer_layout(),
            pipeline_model_parallel_size=plan.pp_size,
        )
        parsed.validate_layer_layout(num_layers=78, mtp_num_layers=0)
        self.assertEqual(
            sum(
                parsed.get_num_layers_to_build(pp_rank=i)
                for i in range(plan.pp_size)
            ), 78
        )


if __name__ == "__main__":
    unittest.main()
