"""Offline checks for the CUDA / dependency preflight; no GPU or HF job required."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.check_glm53_gpu_stack import (
    TRANSFER_REQUIREMENTS, audit, check_cuda_device, check_packages,
)


class GLM53GPUStackTests(unittest.TestCase):
    def good_versions(self):
        return {
            "numpy": "2.1.0",
            "safetensors": "0.6.0",
            "huggingface-hub": "0.35.0",
            "requests": "2.32.3",
            "packaging": "25.0",
            "torch": "2.10.0+cu128",
            "transformers": "5.17.0",
        }

    def test_dependency_contract_accepts_cuda_build_suffix(self):
        self.assertEqual(check_packages(self.good_versions()), [])
        self.assertEqual(check_packages(self.good_versions(), model_runtime=True), [])

    def test_dependency_contract_rejects_missing_or_mismatched(self):
        versions = self.good_versions()
        versions.pop("safetensors")
        versions["torch"] = "2.9.1+cu128"
        versions["requests"] = "3.0.0"
        problems = check_packages(versions)
        self.assertTrue(any("safetensors" in p for p in problems))
        self.assertTrue(any("torch" in p for p in problems))
        self.assertTrue(any("requests" in p for p in problems))

    def test_optional_inference_stack_is_separate_from_transfer(self):
        versions = self.good_versions()
        versions.pop("transformers")
        self.assertEqual(check_packages(versions), [])
        self.assertTrue(check_packages(versions, model_runtime=True))

    def test_cuda_required_fails_closed_on_cpu(self):
        issues = check_cuda_device(
            {"cuda_available": False}, require_cuda=True,
            min_vram_gib=20.0, require_bf16=False,
        )
        self.assertTrue(any("refusing CPU fallback" in s for s in issues))
        self.assertEqual(check_cuda_device(
            {"cuda_available": False}, require_cuda=False,
            min_vram_gib=0, require_bf16=False,
        ), [])

    def test_memory_and_bf16_are_independent_gates(self):
        gpu = {
            "cuda_available": True, "torch_cuda_build": "12.8",
            "sm_major": 7, "vram_gib": 15.5, "bf16_supported": False,
        }
        self.assertEqual(check_cuda_device(
            gpu, require_cuda=True, min_vram_gib=0,
            require_bf16=False,
        ), [])
        issues = check_cuda_device(
            gpu, require_cuda=True, min_vram_gib=20,
            require_bf16=True,
        )
        self.assertEqual(len(issues), 2)

    @patch("scripts.check_glm53_gpu_stack.tool_version", return_value=None)
    @patch("scripts.check_glm53_gpu_stack.cuda_snapshot",
           return_value={"cuda_available": False})
    @patch("scripts.check_glm53_gpu_stack.package_versions")
    def test_full_audit_blocks_non_cuda_worker(self, package_versions, *_):
        package_versions.return_value = self.good_versions()
        report = audit(
            require_cuda=True, min_vram_gib=20.0, require_bf16=False,
            model_runtime=False, smoke=True, pip_check=False,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["issues"])
        self.assertIn("cuda", report)

    def test_transfer_requirements_do_not_require_tilelang_for_compiler(self):
        self.assertNotIn("tilelang", TRANSFER_REQUIREMENTS)
        self.assertNotIn("mamba-ssm", TRANSFER_REQUIREMENTS)


if __name__ == "__main__":
    unittest.main()
