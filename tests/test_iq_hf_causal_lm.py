"""Official Transformers IQ model registration and fail-closed runtime API."""
from __future__ import annotations

import unittest

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from iq_training.hf_causal_lm import (
    IQHFForCausalLM,
    IQHFModelError,
    register_iq_hf_causal_lm,
)
from iq_training.hf_model_config import IQHFCausalConfig


class IQHFModelTests(unittest.TestCase):
    def test_register_real_iq_class_with_transformers(self):
        self.assertTrue(issubclass(IQHFForCausalLM, PreTrainedModel))
        self.assertIs(IQHFForCausalLM.config_class, IQHFCausalConfig)
        self.assertEqual(IQHFForCausalLM.base_model_prefix, "model")
        register_iq_hf_causal_lm()
        mapping = AutoModelForCausalLM._model_mapping
        self.assertIs(mapping[IQHFCausalConfig], IQHFForCausalLM)

    def test_native_mamba3_model_never_falls_back_to_cpu_or_fake_gpt(self):
        if torch.cuda.is_available():
            self.skipTest("host has CUDA; CPU rejection applies only without NVIDIA GPU")
        with self.assertRaisesRegex(IQHFModelError, "NVIDIA CUDA"):
            IQHFForCausalLM(IQHFCausalConfig())

    def test_invalid_config_is_rejected_before_full_model_allocation(self):
        with self.assertRaisesRegex(IQHFModelError, "IQHFCausalConfig"):
            IQHFForCausalLM(object())


if __name__ == "__main__":
    unittest.main()
