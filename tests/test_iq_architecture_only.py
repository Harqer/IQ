"""Architecture-only IQ validation: no teacher, donor or learned weights."""
from __future__ import annotations

from contextlib import chdir, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from iq_model import HybridLayerType, validate_canonical_hybrid_backbone
from iq_model.glm_tokenizer import with_glm_tokenizer_config
from iq_transfer.complete_transplant import canonical_glm53_config
from scripts.audit_iq_architecture_only import (
    ArchitectureOnlyError,
    inspect_architecture_only,
    main,
)


class IQArchitectureOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspection = inspect_architecture_only()

    def test_preserves_all_canonical_iq_operators_and_hidden_width(self):
        result = self.inspection
        config = result.config
        base = canonical_glm53_config()
        self.assertEqual(config.schedule, base.schedule)
        self.assertEqual(config.mamba3, base.mamba3)
        self.assertEqual(config.moe_variant, "stable_latent")
        self.assertEqual(config.stable_moe, base.stable_moe)
        self.assertEqual(config.compressed_context, base.compressed_context)
        self.assertEqual(config.attnres, base.attnres)
        self.assertEqual(config.reasoning, base.reasoning)
        self.assertEqual(config.model.hidden_size, 4096)
        self.assertEqual(config.model.vocab_size, 154880)
        validate_canonical_hybrid_backbone(config)
        self.assertEqual(config.schedule.count(HybridLayerType.MAMBA3), 32)

    def test_glm_tokenizer_geometry_matches_expected_lexical_state(self):
        report = self.inspection.report
        self.assertEqual(report["embedding_weight_shape"], [154880, 4096])
        self.assertEqual(report["lm_head_weight_shape"], [154880, 4096])
        self.assertEqual(report["mimo_rank"], 4)
        self.assertEqual(report["context_layers"], {"csa": 12, "hca": 12})
        self.assertEqual(report["moe_layers"], 24)
        self.assertGreater(report["expected_state_tensor_keys"], 0)
        self.assertFalse(report["donor_neural_weights_loaded"])
        self.assertFalse(report["glm_weight_transfer_executed"])
        self.assertFalse(report["model_forward_executed"])

    def test_no_weight_transfer_is_called(self):
        with (
            patch("iq_transfer.glm53_compile.compile_glm53_iq_checkpoint",
                  side_effect=AssertionError("Unexpected weight compiler invocation")),
            patch("iq_transfer.mamba3_direct.load_official_mamba3_foundation_globals",
                  side_effect=AssertionError("Unexpected neural-weight load")),
        ):
            self.assertEqual(inspect_architecture_only().report["status"],
                             "canonical_iq_architecture_schema_valid")

    def test_command_has_no_required_paths_or_weights(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([]), 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["schedule_unchanged"])
        self.assertNotIn("written_config", report)

    def test_write_config_creates_actual_named_artifact_and_refuses_overwrite(self):
        with TemporaryDirectory() as temp:
            with chdir(temp):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["--write-config"]), 0)
                payload = json.loads(output.getvalue())
                result_path = Path(payload["written_config"])
                self.assertEqual(result_path.name, "iq_glm53_architecture.json")
                self.assertTrue(result_path.is_file())
                config = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual(config["model"]["vocab_size"], 154880)
                self.assertEqual(len(config["schedule"]["layers"]), 80)
                with self.assertRaisesRegex(ArchitectureOnlyError, "refusing overwrite"):
                    main(["--write-config"])

    def test_glm_tokenizer_changes_vocabulary_only(self):
        base = canonical_glm53_config()
        configured = with_glm_tokenizer_config(base)
        self.assertEqual(base.model.vocab_size, 128256)
        self.assertEqual(configured.model.vocab_size, 154880)
        self.assertEqual(
            {k: v for k, v in configured.to_dict().items() if k != "model"},
            {k: v for k, v in base.to_dict().items() if k != "model"},
        )


if __name__ == "__main__":
    unittest.main()
