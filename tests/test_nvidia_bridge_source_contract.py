"""Inspect the *installed NVIDIA 0.6.2 release source* without GPU imports.

Current NVIDIA Bridge __init__ eagerly imports models requiring
transformer_engine.pytorch, so executing its registry on a standard CPU
runner is unsupported. This checks the real wheel's implementation and
reversible DirectMapping contract via AST; never an imitation registry.

Actual runtime tests are a separate CUDA acceptance gate.
"""
from __future__ import annotations

import ast
from importlib.metadata import distribution, version
from pathlib import Path
import unittest


class NvidiaBridgeWheelSourceTests(unittest.TestCase):
    def source(self, path: str) -> ast.Module:
        wheel = distribution("megatron-bridge")
        file = Path(wheel.locate_file("megatron/bridge/models/conversion")) / path
        if not file.is_file():
            self.fail(f"genuine NVIDIA Bridge source file missing: {file}")
        return ast.parse(file.read_text(encoding="utf-8"), filename=str(file))

    def test_pinned_actual_megatron_bridge_release(self):
        self.assertEqual(version("megatron-bridge"), "0.6.2")
        self.assertEqual(version("megatron-core"), "0.19.0")

    def test_actual_direct_mapping_source_contract(self):
        tree = self.source("param_mapping.py")
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.assertIn("MegatronParamMapping", classes)
        self.assertIn("DirectMapping", classes)
        direct = classes["DirectMapping"]
        methods = {
            n.name: n for n in direct.body if isinstance(n, ast.FunctionDef)
        }
        self.assertIn("hf_to_megatron", methods)
        self.assertIn("megatron_to_hf", methods)
        # NVIDIA DirectMapping.hf_to_megatron returns the input weights:
        # exact 1:1 CPU copying is safe ONLY after exhaustive shape/name
        # checks, unlike GLM->IQ operator conversion.
        imports = methods["hf_to_megatron"]
        returned_names = {
            node.value.id
            for node in ast.walk(imports)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
        }
        self.assertIn("hf_weights", returned_names)
        self.assertTrue(any(
            isinstance(node, ast.Attribute)
            and node.attr == "broadcast_from_pp_rank"
            for node in ast.walk(methods["megatron_to_hf"])
        ))

    def test_real_registry_both_lookup_interfaces_exist(self):
        tree = self.source("mapping_registry.py")
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.assertIn("MegatronMappingRegistry", classes)
        method_names = {
            node.name for node in classes["MegatronMappingRegistry"].body
            if isinstance(node, ast.FunctionDef)
        }
        for required in (
            "__init__", "megatron_to_hf_lookup",
            "hf_to_megatron_lookup", "resolve_mapping",
        ):
            self.assertIn(required, method_names)


if __name__ == "__main__":
    unittest.main()
