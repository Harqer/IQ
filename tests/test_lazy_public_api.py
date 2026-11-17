"""CPU-only regression gates for IQ's branch-separated model imports."""
from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("iq_model", "iq_transfer", "iq_training")


class LazyPublicAPITests(unittest.TestCase):
    def test_exports_preserved_and_files_resolve(self):
        for package in PACKAGES:
            with self.subTest(package=package):
                path = ROOT / package / "__init__.py"
                tree = ast.parse(path.read_text(encoding="utf-8"))
                constants = {
                    target.id: ast.literal_eval(stmt.value)
                    for stmt in tree.body
                    if isinstance(stmt, ast.Assign)
                    for target in stmt.targets
                    if isinstance(target, ast.Name)
                    and target.id in ("__all__", "_MODULE_EXPORTS")
                }
                exported = constants["__all__"]
                module_map = constants["_MODULE_EXPORTS"]
                mapped = [name for names in module_map.values() for name in names]
                self.assertTrue(exported)
                self.assertEqual(len(mapped), len(set(mapped)))
                self.assertEqual(set(exported), set(mapped))
                for module in module_map:
                    self.assertTrue(
                        (ROOT / package / f"{module}.py").is_file()
                        or (ROOT / package / module / "__init__.py").is_file(),
                        f"{package}.{module} is missing",
                    )

    def test_importing_top_level_packages_does_not_load_old_donors(self):
        code = '''import sys
import iq_model, iq_training, iq_transfer
assert "iq_transfer.gpt_oss20b" not in sys.modules
assert "iq_transfer.phi4" not in sys.modules
assert "iq_transfer.complete_transplant" not in sys.modules
assert "iq_model.hybrid" not in sys.modules
assert "torch" not in sys.modules
for package in (iq_model, iq_training, iq_transfer):
    assert set(package.__all__).issubset(set(dir(package)))
print("LAZY_IMPORTS_OK")
'''
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LAZY_IMPORTS_OK", result.stdout)

    def test_student_lane_never_imports_legacy_checkpoint_loader(self):
        for relative in (
            "iq_model/glm_mamba3_student.py",
            "scripts/iq_glm53_mamba3.py",
            "iq_training/glm_mamba3_distillation.py",
        ):
            with self.subTest(path=relative):
                tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
                imports = [
                    stmt.module or ""
                    for stmt in ast.walk(tree)
                    if isinstance(stmt, ast.ImportFrom)
                ]
                self.assertFalse(any(
                    "complete_transplant" in name or "gpt_oss20b" in name
                    or "phi_pipeline" in name
                    for name in imports
                ))


if __name__ == "__main__":
    unittest.main()
