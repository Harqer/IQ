"""Real two-rank MCore torch_dist model+AdamW+scheduler+RNG restart test.

Requires actual megatron-core installed. Test is not a simulated checkpoint:
it uses NVIDIA's distributed serialization and two Gloo worker processes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from iq_model.mlp import ExpertParallelStableLatentMoE, StableLatentMoEConfig
from iq_training.mcore_checkpoint import (
    IQCheckpointProgress, IQDistributedCheckpointError,
    load_training_checkpoint, save_training_checkpoint,
)


class _DistributedStudent(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.moe = ExpertParallelStableLatentMoE(
            StableLatentMoEConfig(
                hidden_size=8, latent_size=6, expert_intermediate_size=12,
                num_experts=4, top_k=2, num_shared_experts=1,
                situ_beta=4.0, situ_linear_beta=25.0,
            ), ep_group=group
        )

    def forward(self, x):
        return self.moe(x).hidden_states


def _two_rank_restart(rank: int, world: int, url: str, folder: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=url, rank=rank, world_size=world)
    try:
        torch.manual_seed(405)
        model = _DistributedStudent(dist.group.WORLD)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        torch.manual_seed(2020 + rank)
        x = torch.randn(6 + rank, 8)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        state_before = {key: val.clone() for key, val in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        trained = {
            key: {
                field: data.clone() if isinstance(data, torch.Tensor) else data
                for field, data in optimizer.state.get(param, {}).items()
            }
            for key, param in model.named_parameters()
        }
        progress = IQCheckpointProgress(
            iteration=1, consumed_samples=100 + rank,
            run_id="iq-resume-integration-001",
            config_fingerprint="tiny-ep2-glm-full-router-v1"
        )
        checkpoint = save_training_checkpoint(
            model, optimizer, scheduler=scheduler,
            dataloader_state={"next_sample": 100 + rank},
            progress=progress, base_dir=folder,
        )
        assert checkpoint.joinpath(".metadata").exists()
        assert (Path(folder) / "iq_latest.json").is_file()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(3.0)
        optimizer.param_groups[0]["lr"] = 0.055
        scheduler.step()
        torch.manual_seed(8000 + rank)
        loaded_progress, loader = load_training_checkpoint(
            model, optimizer, scheduler=scheduler,
            base_dir=folder,
            expected_run_id="iq-resume-integration-001",
            expected_fingerprint="tiny-ep2-glm-full-router-v1",
        )
        assert loaded_progress == progress
        assert loader == {"next_sample": 100 + rank}
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert optimizer.param_groups[0]["lr"] == 0.004
        assert scheduler.last_epoch == 1
        for name, actual in model.state_dict().items():
            torch.testing.assert_close(actual, state_before[name], atol=0, rtol=0)
        for name, p in model.named_parameters():
            prior = trained[name]
            restored = optimizer.state.get(p, {})
            assert prior.keys() == restored.keys()
            for key in prior:
                if isinstance(prior[key], torch.Tensor):
                    torch.testing.assert_close(prior[key], restored[key], atol=0, rtol=0)
                else:
                    assert prior[key] == restored[key]

        # A real post-resume optimizer update, not merely tensor comparison.
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()
        assert (sum(p.norm().item() for p in model.parameters())) > 0

        # Existing iterations cannot be silently overwritten.
        try:
            save_training_checkpoint(
                model, optimizer, scheduler=scheduler,
                dataloader_state={"next_sample": 105},
                progress=progress, base_dir=folder,
            )
        except IQDistributedCheckpointError as exc:
            assert "overwrite" in str(exc)
        else:
            raise AssertionError("overwriting native checkpoint was allowed")
    finally:
        dist.destroy_process_group()


class IQMCoreCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("megatron") is not None,
        "install real megatron-core==0.19.0 to run the native checkpoint acceptance gate"
    )
    def test_two_rank_torch_dist_full_training_state_restart(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("Gloo distributed is unavailable")
        with TemporaryDirectory() as folder:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            mp.spawn(
                _two_rank_restart,
                args=(2, f"tcp://127.0.0.1:{port}", folder),
                nprocs=2, join=True,
            )


if __name__ == "__main__":
    unittest.main()
