"""Exercise actual NVIDIA ModelConfig / ModelBuilder interfaces on CPU.

The model is intentionally NOT instantiated on a CPU runner: IQ's genuine
Mamba-3 MIMO path requires CUDA and 78 full expert stages. All unsupported
topologies and wrapper types are rejected before model allocation.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

try:
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.training.models.base import ModelConfig, ModelBuilder
    from iq_training.mcore_builder import (
        IQBridgeBuilderError,
        IQMegatronModelBuilder,
        IQMegatronModelConfig,
        iq_mcore_transformer_metadata,
    )
except ImportError:
    MCORE_PRESENT = False
else:
    MCORE_PRESENT = True

from iq_transfer.complete_transplant import nvidia_parallel_78_config


@unittest.skipUnless(MCORE_PRESENT, "official megatron-core is not installed in CPU unit job")
class IQOfficialMCoreModelBuilderTests(unittest.TestCase):
    def test_real_nvidia_builder_registration_and_serialization(self):
        cfg = IQMegatronModelConfig(
            iq_hybrid=nvidia_parallel_78_config(expert_parallel_size=8).to_dict(),
            expert_model_parallel_size=8,
        )
        self.assertIs(cfg.get_builder_cls(), IQMegatronModelBuilder)
        builder = cfg.get_builder_cls()(cfg)
        self.assertIsInstance(builder, ModelBuilder)
        serialized = cfg.as_dict()
        self.assertEqual(serialized["_builder_"], IQMegatronModelConfig.builder)
        self.assertEqual(serialized["iq_hybrid"], cfg.iq_hybrid)
        restored = ModelConfig.from_dict(serialized)
        self.assertIsInstance(restored, IQMegatronModelConfig)
        self.assertEqual(
            restored.validated_iq_config().fingerprint,
            cfg.validated_iq_config().fingerprint,
        )
        self.assertEqual(restored.expert_model_parallel_size, 8)

    def test_real_transformer_config_matches_custom_iq_geometry(self):
        iq = nvidia_parallel_78_config(expert_parallel_size=8)
        native = iq_mcore_transformer_metadata(iq)
        self.assertEqual(native.num_layers, 78)
        self.assertEqual(native.hidden_size, 4096)
        self.assertEqual(native.ffn_hidden_size, 12288)
        self.assertEqual(native.num_moe_experts, 256)
        self.assertEqual(native.moe_router_topk, 8)
        self.assertEqual(native.expert_model_parallel_size, 8)
        self.assertEqual(native.tensor_model_parallel_size, 1)
        self.assertEqual(native.pipeline_model_parallel_size, 1)

    def test_rejects_tp_pp_vpp_and_inconsistent_expert_groups(self):
        iq = nvidia_parallel_78_config(expert_parallel_size=8)
        base = dict(iq_hybrid=iq.to_dict(), expert_model_parallel_size=8)
        for option, error in (
            (dict(tensor_model_parallel_size=2), "TP must equal 1"),
            (dict(pipeline_model_parallel_size=2), "PP must equal 1"),
            (dict(virtual_pipeline_model_parallel_size=2), "virtual pipeline"),
            (dict(expert_model_parallel_size=2), "contradicts"),
            (dict(params_dtype="float32"), "BF16"),
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(IQBridgeBuilderError, error):
                    IQMegatronModelConfig(**{**base, **option})

    def test_rejects_unsupported_wrappers_before_cuda_allocation(self):
        cfg = IQMegatronModelConfig()
        builder = IQMegatronModelBuilder(cfg)
        with self.assertRaisesRegex(IQBridgeBuilderError, "not implemented"):
            builder.build_distributed_models(
                pg_collection=ProcessGroupCollection(), wrap_with_ddp=True,
            )
        with self.assertRaisesRegex(IQBridgeBuilderError, "initialize real torch.distributed"):
            builder.build_model(ProcessGroupCollection())


if __name__ == "__main__":
    unittest.main()
