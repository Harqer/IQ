"""CPU tests for exact BF16 raw donor expert export without attention reads."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from iq_transfer.complete_transplant import canonical_glm53_config
from iq_transfer.glm53_raw_expert_export import (
    RawExpertExportError,
    expert_keys,
    export_raw_selected_experts,
    validate_export_plan,
)
from scripts.probe_glm53_remote_ranges import REVISION


class FakeSource:
    def __init__(self, tensors):
        self.tensors = tensors
        self.calls = []

    def shape(self, key):
        self.calls.append(("shape", key))
        return self.tensors[key].shape

    def get(self, key):
        self.calls.append(("get", key))
        return self.tensors[key]


def fixture():
    hidden, intermediate = 4, 3
    prefix = "model.layers.3.mlp"
    data = {
        f"{prefix}.gate.weight": torch.arange(
            256 * hidden, dtype=torch.float32
        ).reshape(256, hidden).to(torch.bfloat16),
        f"{prefix}.gate.e_score_correction_bias": torch.arange(
            256, dtype=torch.float32
        ).to(torch.bfloat16),
    }
    for group in (f"{prefix}.shared_experts", f"{prefix}.experts.7"):
        for index, (key, shape) in enumerate(zip(
            expert_keys(group),
            ((intermediate, hidden), (intermediate, hidden),
             (hidden, intermediate)),
            strict=True,
        )):
            data[key] = torch.arange(
                shape[0] * shape[1], dtype=torch.float32
            ).reshape(shape).to(torch.bfloat16) + index
    return data


def example_plan():
    stages = [0] + list(range(3, 26))
    ids = [None] + [list(range(64)) for _ in range(23)]
    path = "model-00001-of-00282.safetensors"
    index = {}
    for layer in stages[1:]:
        root = f"model.layers.{layer}.mlp"
        names = [f"{root}.gate.weight",
                 f"{root}.gate.e_score_correction_bias",
                 *expert_keys(f"{root}.shared_experts")]
        for expert in range(64):
            names.extend(expert_keys(f"{root}.experts.{expert}"))
        index.update({key: path for key in names})
    plan = {
        "schema_version": 1,
        "status": "selective_metadata_plan_only",
        "source_revision": REVISION,
        "target_experts": 64,
        "target_config_fingerprint": canonical_glm53_config(
            routed_experts=64
        ).fingerprint,
        "official_total_shards": 282,
        "source_layers": stages,
        "selected_donor_expert_ids": ids,
        "official_shards": [path],
    }
    return plan, index


class RawDonorExpertExportTests(unittest.TestCase):
    def test_pinned_plan_selects_exact_23_sparse_stages(self):
        plan, index = example_plan()
        tasks = validate_export_plan(plan, index)
        self.assertEqual(len(tasks), 23)
        self.assertEqual(tasks[0], (1, 3, tuple(range(64))))
        self.assertEqual(tasks[-1][1], 25)

    def test_plan_with_wrong_revision_fails_before_download(self):
        plan, index = example_plan()
        plan["source_revision"] = "unapproved"
        with self.assertRaisesRegex(RawExpertExportError, "revision"):
            validate_export_plan(plan, index)

    def test_missing_selected_expert_tensor_fails_before_download(self):
        plan, index = example_plan()
        del index["model.layers.3.mlp.experts.1.gate_proj.weight"]
        with self.assertRaisesRegex(RawExpertExportError, "missing"):
            validate_export_plan(plan, index)

    def test_smoke_export_retains_every_bf16_bit_and_excludes_attention(self):
        data = fixture()
        source = FakeSource(data)
        plan, _ = example_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw-only"
            report = export_raw_selected_experts(
                source=source, plan=plan,
                tasks=[(1, 3, (7, 8))], root=root,
                hidden=4, intermediate=3,
                index_sha256="abc",
                max_stages=1, max_experts=1,
            )
            self.assertEqual(report["status"], "partial_smoke_only")
            self.assertEqual(report["tensor_count"], 8)
            self.assertTrue(report["not_an_iq_checkpoint"])
            self.assertFalse((root / "model.safetensors.index.json").exists())
            self.assertFalse((root / "iq_config.json").exists())
            self.assertFalse((root / "INCOMPLETE").exists())
            self.assertTrue((root / "raw_expert_manifest.json").is_file())
            for key, shard in report["weight_map"].items():
                value = load_file(str(root / shard))[key]
                self.assertEqual(value.dtype, torch.bfloat16)
                want_sha = hashlib.sha256(
                    data[key].contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                self.assertEqual(report["tensor_sha256"][key], want_sha)
                self.assertTrue(torch.equal(value.view(torch.uint8),
                                            data[key].view(torch.uint8)))
            self.assertFalse(any("self_attn" in key for _, key in source.calls))
            self.assertFalse(any(".experts.8." in key for _, key in source.calls))

    def test_router_bias_fp32_is_preserved_without_cast(self):
        tensors = fixture()
        bias = "model.layers.3.mlp.gate.e_score_correction_bias"
        tensors[bias] = tensors[bias].float()
        source = FakeSource(tensors)
        plan, _ = example_plan()
        with tempfile.TemporaryDirectory() as tmp:
            report = export_raw_selected_experts(
                source=source, plan=plan,
                tasks=[(1, 3, (7,))], root=Path(tmp) / "fp32",
                hidden=4, intermediate=3,
                index_sha256="abc", max_experts=1,
            )
            self.assertEqual(report["tensor_dtype"][bias], "float32")
            filename = report["weight_map"][bias]
            got = load_file(str(Path(tmp) / "fp32" / filename))[bias]
            self.assertEqual(got.dtype, torch.float32)
            self.assertTrue(torch.equal(got, tensors[bias]))

    def test_reject_output_overwrite_and_nonbf16_payload(self):
        plan, _ = example_plan()
        source = FakeSource(fixture())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "occupied"
            root.mkdir()
            (root / "important.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(RawExpertExportError, "NEW empty"):
                export_raw_selected_experts(
                    source=source, plan=plan,
                    tasks=[(1, 3, (7,))], root=root,
                    hidden=4, intermediate=3, index_sha256="abc",
                )
            self.assertEqual((root / "important.txt").read_text(), "preserve")
            root2 = Path(tmp) / "bad"
            key = "model.layers.3.mlp.gate.weight"
            source.tensors[key] = source.tensors[key].float()
            with self.assertRaisesRegex(RawExpertExportError, "invalid shape, device or dtype"):
                export_raw_selected_experts(
                    source=source, plan=plan,
                    tasks=[(1, 3, (7,))], root=root2,
                    hidden=4, intermediate=3, index_sha256="abc",
                )
            self.assertTrue((root2 / "INCOMPLETE").exists())
            self.assertFalse((root2 / "raw_expert_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
