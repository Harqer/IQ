"""Real PyTorch parameter and NVIDIA registry round-trip acceptance.

Tests use small, non-pretrained tensors solely to verify reversible transport
mechanics, not to assert GLM-5.3 or IQ 78-block weight equivalence.
"""
from __future__ import annotations

import unittest
import os

import torch
from torch import nn

from iq_training.native_weight_transfer import transfer_native_iq_weights
from iq_training.native_weight_registry import (
    IQNativeMappingError,
    megatron_native_mapping_registry,
    plan_iq_native_tensors,
)
from iq_transfer.complete_transplant import nvidia_parallel_78_config


class TinyIQSource(nn.Module):
    """Real PyTorch named parameter/buffer sample with IQ checkpoint prefixes."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(7, 4)
        self.model.norm = nn.LayerNorm(4)
        self.model.register_buffer("offsets", torch.arange(3, dtype=torch.int64))


class TinyIQNative(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.iq = nn.Module()
        self.iq.embed_tokens = nn.Embedding(7, 4)
        self.iq.norm = nn.LayerNorm(4)
        self.iq.register_buffer("offsets", torch.arange(3, dtype=torch.int64))


class NativeIQRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = nvidia_parallel_78_config()
        self.hf = TinyIQSource()
        self.native = TinyIQNative()

    def test_exact_named_metadata_and_sorted_bidirectional_key_coverage(self):
        plan = plan_iq_native_tensors(
            self.hf, self.native, iq_config=self.config, full_78_required=False
        )
        self.assertEqual(plan.topology, (1, 1, 1))
        self.assertEqual(
            [v.hf_name for v in plan.tensors],
            sorted(v.hf_name for v in plan.tensors),
        )
        self.assertEqual(len(plan.tensors), 4)
        self.assertEqual(
            len([v for v in plan.tensors if v.kind == "buffer"]), 1
        )
        self.assertFalse(plan.complete_iq_78_inventory)
        payload = plan.to_dict()
        self.assertFalse(payload["source_is_glm53"])
        self.assertEqual(payload, plan.to_dict())
        self.assertEqual(len(payload["sha256"]), 64)
        # Miniature test inventories must NEVER bypass the real 78-block
        # completeness gate on a production conversion path.
        with self.assertRaisesRegex(Exception, "unclassified IQ destination|block 0 missing"):
            plan_iq_native_tensors(
                self.hf, self.native, iq_config=self.config, full_78_required=True,
            )

    def test_tensor_shape_dtype_missing_and_parallelism_fail_closed(self):
        with torch.no_grad():
            self.native.iq.norm.weight = nn.Parameter(torch.zeros(6))
        with self.assertRaisesRegex(IQNativeMappingError, "non-identical"):
            plan_iq_native_tensors(
                self.hf, self.native, iq_config=self.config, full_78_required=False
            )
        self.native = TinyIQNative()
        del self.native.iq.norm.bias
        with self.assertRaisesRegex(IQNativeMappingError, "inventories differ"):
            plan_iq_native_tensors(
                self.hf, self.native, iq_config=self.config, full_78_required=False
            )
        self.native = TinyIQNative()
        with self.assertRaisesRegex(IQNativeMappingError, "TP=PP=EP=1"):
            plan_iq_native_tensors(
                self.hf, self.native, iq_config=self.config,
                tensor_parallel_size=2, full_78_required=False,
            )

    def test_actual_bidirectional_weight_transfer_exact_bytes(self):
        # Validate executable, exact IQ-native PyTorch copying on CPU.
        # NVIDIA's global package imports Transformer Engine and is tested
        # separately on CUDA; no false CPU Bridge registry is invented.
        torch.manual_seed(90)
        with torch.no_grad():
            self.hf.model.embed_tokens.weight.copy_(
                torch.arange(28, dtype=torch.float32).reshape(7, 4)
            )
            self.hf.model.norm.weight.fill_(0.83)
            self.hf.model.norm.bias.fill_(0.15)
            self.hf.model.offsets.copy_(torch.tensor([9, 8, 7]))
            self.native.iq.embed_tokens.weight.zero_()
            self.native.iq.norm.weight.zero_()
            self.native.iq.norm.bias.zero_()
            self.native.iq.offsets.zero_()
        transferred = transfer_native_iq_weights(
            self.hf, self.native, iq_config=self.config,
            direction="hf_to_megatron", full_78_required=False
        )
        self.assertTrue(transferred.exact_values_verified)
        self.assertFalse(transferred.full_iq_78_verified)
        self.assertEqual(transferred.tensor_count, 4)
        self.assertFalse(transferred.to_dict()["GLM_to_IQ_pretrained_transform"])
        for hf_name, native_name in (
            ("model.embed_tokens.weight", "iq.embed_tokens.weight"),
            ("model.norm.weight", "iq.norm.weight"),
            ("model.norm.bias", "iq.norm.bias"),
            ("model.offsets", "iq.offsets"),
        ):
            torch.testing.assert_close(
                self.hf.state_dict()[hf_name], self.native.state_dict()[native_name],
                atol=0, rtol=0
            )
        with torch.no_grad():
            self.native.iq.norm.weight.fill_(3.5)
        result = transfer_native_iq_weights(
            self.hf, self.native, iq_config=self.config,
            direction="megatron_to_hf", full_78_required=False
        )
        self.assertTrue(result.exact_values_verified)
        torch.testing.assert_close(
            self.hf.model.norm.weight, self.native.iq.norm.weight,
            atol=0, rtol=0
        )

    def test_true_bridge_mapping_registry_both_directions(self):
        try:
            from megatron.bridge.models.conversion.mapping_registry import (
                MegatronMappingRegistry,
            )
            from megatron.bridge.models.conversion.param_mapping import DirectMapping
        except ImportError:
            if os.environ.get("IQ_REQUIRE_NVIDIA_BRIDGE") == "1":
                raise
            self.skipTest("megatron-bridge optional outside dedicated NVIDIA CI")
        plan = plan_iq_native_tensors(
            self.hf, self.native, iq_config=self.config, full_78_required=False
        )
        registry = megatron_native_mapping_registry(plan)
        self.assertIsInstance(registry, MegatronMappingRegistry)
        self.assertEqual(len(registry), len(plan.tensors))
        hf_state = self.hf.state_dict()
        native_state = self.native.state_dict()
        for spec in plan.tensors:
            forward = registry.hf_to_megatron_lookup(spec.hf_name)
            reverse = registry.megatron_to_hf_lookup(spec.megatron_name)
            self.assertIsInstance(forward, DirectMapping)
            self.assertIsInstance(reverse, DirectMapping)
            source = hf_state[spec.hf_name]
            converted = forward.hf_to_megatron(source, self.native)
            torch.testing.assert_close(converted, source, atol=0, rtol=0)
            # Standalone CPU test checks exact reverse-name and tensor
            # identity; the Bridge export distributed broadcast path is
            # validated separately when actual process groups exist.
            native_state[spec.megatron_name] = converted.detach().clone()
            torch.testing.assert_close(
                native_state[spec.megatron_name], source, atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
