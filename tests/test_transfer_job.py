from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn

from iq_model import DoRALinear, IQForCausalLM, IQModelConfig
from iq_transfer import DonorRuntime, MappingTensorSource, Phi4Inspector, build_donor_manifest
from iq_transfer.batches import load_token_batches, save_token_batches
from iq_transfer.calibration import CalibrationManifest, CalibrationRecord, CalibrationSplit
from iq_transfer.job import load_transferred_iq_artifact, run_phi_dense_transfer_loaded


class FakePhiAttention(nn.Module):
    def __init__(self, hidden=8, q_heads=4, kv_heads=2, head_dim=2):
        super().__init__()
        self.q_width = q_heads * head_dim
        self.kv_width = kv_heads * head_dim
        self.qkv_proj = nn.Linear(hidden, self.q_width + 2 * self.kv_width, bias=False)
        self.o_proj = nn.Linear(self.q_width, hidden, bias=False)

    def forward(self, x, **kwargs):
        qkv = self.qkv_proj(x)
        return self.o_proj(qkv[..., : self.q_width]), None


class FakePhiMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class FakePhiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = FakePhiAttention()
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = FakePhiMLP()

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x))[0]
        return x + self.mlp(self.post_attention_layernorm(x))


class FakePhiBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([FakePhiLayer()])
        self.norm = nn.LayerNorm(8)


class FakePhi(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakePhiBase()
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.config = types.SimpleNamespace(
            model_type="phi3",
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            vocab_size=32,
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return types.SimpleNamespace(logits=self.lm_head(x))


def runtime_from_model(phi: FakePhi) -> DonorRuntime:
    cfg = {
        "_name_or_path": "fake-phi",
        "model_type": "phi3",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
    }
    layer = phi.model.layers[0]
    tensors = {
        "model.layers.0.self_attn.qkv_proj.weight": layer.self_attn.qkv_proj.weight.detach().numpy(),
        "model.layers.0.self_attn.o_proj.weight": layer.self_attn.o_proj.weight.detach().numpy(),
        "model.layers.0.mlp.gate_up_proj.weight": layer.mlp.gate_up_proj.weight.detach().numpy(),
        "model.layers.0.mlp.down_proj.weight": layer.mlp.down_proj.weight.detach().numpy(),
        "model.embed_tokens.weight": phi.model.embed_tokens.weight.detach().numpy(),
        "lm_head.weight": phi.lm_head.weight.detach().numpy(),
    }
    source = MappingTensorSource(tensors)
    inspector = Phi4Inspector.from_config_mapping(cfg)
    manifest = build_donor_manifest(
        inspector.config,
        source,
        donor_id="phi",
        checkpoint_revision="test",
        tokenizer_hash="tok",
        license="test",
        operator_layout_version="phi3-fused-v1",
        source_uri="memory://phi",
        allow_metadata_only=True,
    )
    return DonorRuntime("phi", manifest, inspector, source)


class TransferJobTests(unittest.TestCase):
    def test_token_batch_artifact_round_trip(self):
        batches = [
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
            {"input_ids": torch.tensor([[4, 5, 6]])},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "batches"
            save_token_batches(batches, base)
            loaded = load_token_batches(base)
        self.assertEqual(len(loaded.batches), 2)
        self.assertTrue(
            torch.equal(loaded.batches[0]["input_ids"], batches[0]["input_ids"])
        )

    def test_loaded_job_builds_training_ready_artifact_and_round_trips(self):
        torch.manual_seed(9)
        phi = FakePhi()
        runtime = runtime_from_model(phi)
        cfg = IQModelConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=12,
            max_position_embeddings=32,
            rope_theta=10000.0,
        )
        iq = IQForCausalLM(cfg)
        manifest = CalibrationManifest(
            "tok",
            (
                CalibrationRecord("fit", CalibrationSplit.MAP_FIT, "code", "a", "x"),
                CalibrationRecord("val", CalibrationSplit.MAP_VALIDATION, "code", "b", "x"),
                CalibrationRecord(
                    "test", CalibrationSplit.TRANSFER_VALIDATION, "code", "c", "x"
                ),
            ),
        )
        fit = [
            {
                "input_ids": torch.tensor(
                    [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
                )
            }
        ]
        validation = [
            {
                "input_ids": torch.tensor(
                    [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]
                )
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "artifact"
            result = run_phi_dense_transfer_loaded(
                phi_model=phi,
                iq_model=iq,
                donor_runtime=runtime,
                calibration_manifest=manifest,
                fit_batches=fit,
                validation_batches=validation,
                output_dir=out,
                dora_rank=2,
                ridge=1e-3,
                shadow_measurements=16,
            )
            self.assertTrue((out / "COMPLETE").is_file())
            self.assertTrue((out / "bundle_manifest.json").is_file())
            self.assertEqual(len(result.dora_paths), 7)
            self.assertIsInstance(iq.blocks[0].attn.q_proj, DoRALinear)

            before = iq(torch.tensor([[1, 2, 3]])).logits.detach()
            restored = load_transferred_iq_artifact(out)
            after = restored(torch.tensor([[1, 2, 3]])).logits.detach()
            self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-5))

            current = {name for name, _ in restored.named_parameters()}
            recorded = {record.target_parameter for record in result.provenance.records()}
            self.assertEqual(current, recorded)


if __name__ == "__main__":
    unittest.main()
