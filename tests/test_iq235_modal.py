"""No-GPU regression contract for Modal's full IQ probe.

Run this locally before enabling any Modal GPU request. These tests inspect
the actual source text without importing Modal or initiating image builds.
"""
from __future__ import annotations

from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "modal_iq235_phase_a.py"


class ModalIQ235SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_compiles_without_connecting_to_modal(self):
        compile(self.source, str(APP), "exec")

    def test_one_full_235_layer_function_not_student(self):
        self.assertIn("load_phase_a()", self.source)
        self.assertIn("generate_probe(", self.source)
        self.assertIn('"physical_layers"', self.source)
        self.assertIn("== 235", self.source)
        self.assertIn('"glm_tokenizer_vocab"', self.source)
        self.assertIn("== 154880", self.source)
        self.assertNotIn("load_glm_student(", self.source)
        self.assertNotIn("load_complete_iq_checkpoint(", self.source)

    def test_hardware_and_cost_guards(self):
        self.assertIn('gpu="H200:8"', self.source)
        self.assertIn("timeout=480", self.source)
        self.assertIn("startup_timeout=180", self.source)
        self.assertIn("max_containers=1", self.source)
        self.assertIn("min_containers=0", self.source)
        self.assertIn("single_use_containers=True", self.source)
        self.assertIn("scaledown_window=2", self.source)
        self.assertIn("retries=0", self.source)
        self.assertNotIn("@modal.asgi_app", self.source)
        self.assertNotIn("modal.deploy(", self.source)
        self.assertNotIn("detach=True", self.source)

    def test_mamba_pinned_vcs_metadata_and_kernel_tail_guard(self):
        """Reject unpinned local installs that lack pip VCS provenance."""
        self.assertIn(
            "git+https://github.com/state-spaces/mamba.git@"
            "e9594ce1c732d97440f0332fdc43170a2294dbfa",
            self.source,
        )
        self.assertIn("--patch-installed", self.source)
        self.assertIn("--check-installed", self.source)
        self.assertNotIn(
            "pip install --no-build-isolation --no-deps /tmp/iq-mamba3-src",
            self.source,
        )

    def test_source_staging_is_scoped(self):
        self.assertIn("_iq235_baseten_packages", self.source)
        self.assertIn("stage_manifest.json", self.source)
        self.assertIn(".add_local_dir(STAGED,", self.source)
        self.assertNotIn('.add_local_dir(REPO,', self.source)
        self.assertIn('"IQ_PHASE" =', self.source.replace("os.environ[", "").replace("]", " ="))
        
    def test_failure_does_not_fabricate_success(self):
        self.assertIn("all_layers_executed", self.source)
        self.assertIn("Refusing to overwrite", self.source)
        self.assertIn("FULL_IQ235_MODAL_FORWARD_PASS", self.source)


if __name__ == "__main__":
    unittest.main()
