#!/usr/bin/env python3
"""Run one REAL full78 IQ MoE layer using only its 8 selected experts.

Read-only HF checkpoint access. Uses real saved common, shared and selected
expert safetensors weights and the existing StableLatentMoELayer.forward.
Never creates the 256 full routed expert weights or loads the whole model.
CPU-only machines refuse: use a local CUDA GPU for the integration test.

This is an isolated MoE layer test, NOT whole-IQ language generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors.torch import load_file

from iq_model.mlp.latent_moe import (
    SiTUExpert, StableLatentMoEConfig, StableLatentMoELayer,
)

BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "glm53-full78-local-20261025a"
PREFIX = f"IQ/runs/{RUN}"
STAGE = 3
MAX_SHARD_BYTES = 80 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2


def remote_sizes(names: set[str]) -> dict[str, int]:
    from huggingface_hub import get_bucket_paths_info

    paths = [f"{PREFIX}/{name}" for name in sorted(names)]
    info = {item.path: item.size for item in get_bucket_paths_info(BUCKET, paths)}
    missing = [name for name in sorted(names) if f"{PREFIX}/{name}" not in info]
    if missing:
        raise RuntimeError(f"Missing remote checkpoint artifacts: {missing}")
    sizes = {name: info[f"{PREFIX}/{name}"] for name in names}
    for name, size in sizes.items():
        if not 0 < size <= MAX_SHARD_BYTES:
            raise RuntimeError(f"Unsafe/oversize single-shard read: {name} ({size} bytes)")
    return sizes


def fetch_to_cache(names: set[str], cache: Path, *, max_total: int = MAX_TOTAL_BYTES) -> dict[str, Path]:
    """Reuse correct-size local shards; write new downloads atomically.

    No bucket mutations; never allow local partial downloads to become
    trusted cache entries. Shard keys/shapes are subsequently checked.
    """
    from huggingface_hub import download_bucket_files

    sizes = remote_sizes(names)
    if sum(sizes.values()) > max_total:
        raise RuntimeError("Planned download exceeds 512 MiB budget")
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in sorted(names):
        if name != Path(name).name or ".." in name or "/" in name or "\\" in name:
            raise RuntimeError("Unsafe checkpoint filename")
        dest = cache / name
        if dest.is_symlink():
            raise RuntimeError(f"Refusing symlinked shard cache {dest}")
        if dest.is_file() and dest.stat().st_size == sizes[name]:
            result[name] = dest
            continue
        if dest.exists():
            raise RuntimeError(f"Unexpected cached file/size: {dest}; clear it manually")
        part = cache / f".{name}.{os.getpid()}.partial"
        try:
            download_bucket_files(
                BUCKET, files=[(f"{PREFIX}/{name}", str(part))],
                raise_on_missing_files=True,
            )
            if not part.is_file() or part.stat().st_size != sizes[name]:
                raise RuntimeError(f"Incomplete remote download: {name}")
            os.replace(part, dest)
        finally:
            part.unlink(missing_ok=True)
        result[name] = dest
        print(f"IQ_REAL_MOE_DOWNLOADED={name} bytes={sizes[name]}", flush=True)
    return result


def assign_tensors(module: nn.Module, tensors: dict[str, torch.Tensor],
                   expected_keys: set[str], *, device: torch.device,
                   prefix: str) -> None:
    if set(tensors) != expected_keys:
        raise RuntimeError(
            f"Wrong saved tensor keys for {prefix}: "
            f"missing={sorted(expected_keys - set(tensors))} "
            f"unexpected={sorted(set(tensors) - expected_keys)}"
        )
    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    with torch.no_grad():
        for full_key, value in tensors.items():
            if not full_key.startswith(prefix):
                raise RuntimeError(f"Wrong tensor prefix {full_key}")
            key = full_key[len(prefix):]
            target = parameters.get(key)
            is_buffer = False
            if target is None:
                target = buffers.get(key)
                is_buffer = True
            if target is None or tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(f"Model geometry mismatch: {full_key}")
            if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Invalid saved tensor numerical data: {full_key}")
            materialized = value.to(device=device, dtype=torch.float32)
            parent_name, _, leaf = key.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            if is_buffer:
                setattr(parent, leaf, materialized)
            else:
                setattr(parent, leaf, nn.Parameter(materialized, requires_grad=False))


def create_sparse_layer(config: StableLatentMoEConfig, *,
                        eps: float, device: torch.device) -> StableLatentMoELayer:
    """Construct production layer on meta then retain only selected experts.

    Identity placeholders are never invoked; after routing we substitute
    every selected index with its real SiTUExpert. The original layer's
    forward and router logic are unchanged.
    """
    with torch.device("meta"):
        layer = StableLatentMoELayer(config, norm_eps=eps, residual_dropout=0.0)
    layer.moe.routed_experts = nn.ModuleList(nn.Identity() for _ in range(config.num_experts))
    layer.moe.shared_experts = nn.ModuleList(nn.Identity() for _ in range(config.num_shared_experts))
    return layer.eval()


def load_expert(tensors: dict[str, torch.Tensor], *,
                config: StableLatentMoEConfig, prefix: str,
                width: int, device: torch.device) -> SiTUExpert:
    with torch.device("meta"):
        module = SiTUExpert(
            width, config.expert_intermediate_size,
            beta=config.situ_beta, linear_beta=config.situ_linear_beta,
            bias=config.expert_bias,
        )
    keys = {f"{prefix}{k}_proj.weight" for k in ("gate", "up", "down")}
    assign_tensors(module, tensors, keys, device=device, prefix=prefix)
    return module.eval()


def assert_fully_materialized(module: nn.Module) -> None:
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        if value.is_meta:
            raise RuntimeError(f"Meta tensor remains in live execution path: {name}")


class ZeroExpert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


@torch.inference_mode()
def run_layer(config_json: dict, common: dict[str, torch.Tensor],
              shared: dict[str, torch.Tensor],
              selected_weights: dict[int, dict[str, torch.Tensor]],
              *, physical: int, seed: int, device: torch.device) -> dict:
    stable = StableLatentMoEConfig(**config_json["stable_moe"])
    if stable.num_experts != 256 or stable.top_k != 8 or stable.num_shared_experts != 1:
        raise RuntimeError("Unexpected full78 MoE routing configuration")
    cfg = config_json["model"]
    layer = create_sparse_layer(stable, eps=float(cfg["rms_norm_eps"]), device=device)
    base = f"layers.{physical}."
    expected_common = {
        base + "norm.weight",
        base + "moe.latent_norm.weight",
        base + "moe.latent_down.weight",
        base + "moe.latent_up.weight",
        base + "moe.router.weight",
        base + "moe.routing_bias",
    }
    assign_tensors(layer, common, expected_common, device=device, prefix=base)
    shared_prefix = base + "moe.shared_experts.0."
    layer.moe.shared_experts[0] = load_expert(
        shared, config=stable, prefix=shared_prefix,
        width=stable.hidden_size, device=device,
    )

    # Fixed seed for reproducibility; this input is synthetic activations,
    # but the router, projections and expert weights are real and persisted.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 1, stable.hidden_size, generator=generator).to(device)
    normed = layer.norm(x)
    raw, selected, weights = layer.moe._route(normed.reshape(-1, stable.hidden_size))
    ids = set(int(n) for n in selected.reshape(-1).tolist())
    if ids != set(selected_weights) or len(ids) != stable.top_k:
        raise RuntimeError(
            f"Selected experts differ from downloaded experts: "
            f"actual={sorted(ids)} provided={sorted(selected_weights)}"
        )
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise RuntimeError("Nonfinite real router scores")
    if not torch.allclose(
        weights.sum(-1),
        torch.full_like(weights.sum(-1), stable.routed_scaling_factor),
        atol=1e-5,
    ):
        raise RuntimeError("Real router selected weights do not normalize")

    for slot, tensors in sorted(selected_weights.items()):
        prefix = base + f"moe.routed_experts.{slot}."
        layer.moe.routed_experts[slot] = load_expert(
            tensors, config=stable, prefix=prefix,
            width=stable.latent_size, device=device,
        )
    assert_fully_materialized(layer)

    output = layer(x)
    if (output.hidden_states.shape != x.shape or
        not bool(torch.isfinite(output.hidden_states).all()) or
        not torch.equal(output.selected_experts.reshape(-1), selected.reshape(-1)) or
        int(output.expert_counts.sum()) != stable.top_k):
        raise RuntimeError("Saved-weight MoE forward output/routing is invalid")
    routed_delta = float((output.hidden_states - x).float().norm().item())
    if routed_delta == 0 or not torch.isfinite(torch.tensor(routed_delta)):
        raise RuntimeError("Real expert MoE produced no finite residual contribution")

    # A real selected expert must affect execution; removing it should
    # change the output while preserving all other saved parameters.
    strongest = int(selected[0, weights[0].argmax()].item())
    real_expert = layer.moe.routed_experts[strongest]
    layer.moe.routed_experts[strongest] = ZeroExpert()
    try:
        ablated = layer(x)
    finally:
        layer.moe.routed_experts[strongest] = real_expert
    max_ablation_delta = float(
        (output.hidden_states - ablated.hidden_states).float().abs().max().item()
    )
    if not 0 < max_ablation_delta < float("inf"):
        raise RuntimeError("Selected expert removal did not affect saved-weight MoE output")
    stats = {
        "stage": STAGE, "physical_layer": physical,
        "selected_experts": sorted(ids),
        "top_k": stable.top_k,
        "routing_scale": stable.routed_scaling_factor,
        "output_norm": float(output.hidden_states.float().norm().item()),
        "moe_residual_norm": routed_delta,
        "selected_expert_ablation_slot": strongest,
        "max_ablation_difference": max_ablation_delta,
        "device": str(device), "compute_dtype": "FP32",
        "checkpoint_weight_dtype": "BF16",
    }
    print("IQ_REAL_CHECKPOINT_MOE_FORWARD_PASS=" + json.dumps(stats, sort_keys=True), flush=True)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default=Path.home() / ".cache/iq-real-moe-stage003")
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--stage", type=int, choices=(3,), default=STAGE,
                   help="First sparse stage; intentionally fixed to keep GPU/download budgets bounded")
    args = p.parse_args()
    if not os.environ.get("HF_TOKEN"):
        p.error("HF_TOKEN missing; run export HF_TOKEN=\"$(hf auth token)\"")
    if not torch.cuda.is_available():
        p.error("CUDA required for real local MoE GPU forward test")
    device = torch.device("cuda:0")
    # Turing RTX 2060 SUPER does not support native BF16 matrix ops.
    # FP32 preserves all BF16 stored weight values exactly and works on CUDA.
    if torch.cuda.get_device_properties(device).total_memory < 4 * 1024**3:
        p.error("Minimum 4 GiB CUDA VRAM needed for bounded FP32 test")

    meta = {"iq_config.json", "COMPILE_COMPLETE.json",
            f"model-moe-{STAGE:03d}-projection.json"}
    artifacts = fetch_to_cache(meta, args.cache_dir)
    config = json.loads(artifacts["iq_config.json"].read_text())
    complete = json.loads(artifacts["COMPILE_COMPLETE.json"].read_text())
    report = json.loads(artifacts[f"model-moe-{STAGE:03d}-projection.json"].read_text())
    if (complete.get("status") != "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED" or
        config["model"]["hidden_size"] != 4096 or
        report.get("stage") != STAGE or report.get("source_layer") != STAGE):
        raise RuntimeError("Wrong full78 checkpoint metadata")
    physical = int(report["target_layer"])
    if config["schedule"]["layers"][physical] != "moe":
        raise RuntimeError("Stage does not map to a physical MoE layer")
    common_name = f"model-moe-{STAGE:03d}-common.safetensors"
    shared_name = f"model-moe-{STAGE:03d}-shared.safetensors"
    initial = fetch_to_cache({common_name, shared_name}, args.cache_dir)
    prefix = f"layers.{physical}."
    common = load_file(str(initial[common_name]), device="cpu")
    shared = load_file(str(initial[shared_name]), device="cpu")

    # Compute selection with the actual saved router, BEFORE downloading any
    # expert shard, so only the selected 8 out of 256 are transferred.
    stable = StableLatentMoEConfig(**config["stable_moe"])
    layer = create_sparse_layer(stable, eps=config["model"]["rms_norm_eps"], device=device)
    assign_tensors(layer, common, {
        prefix + "norm.weight", prefix + "moe.latent_norm.weight",
        prefix + "moe.latent_down.weight", prefix + "moe.latent_up.weight",
        prefix + "moe.router.weight", prefix + "moe.routing_bias",
    }, device=device, prefix=prefix)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(1, 1, stable.hidden_size, generator=generator).to(device)
    _, selected, _ = layer.moe._route(
        layer.norm(x).reshape(-1, stable.hidden_size)
    )
    ids = set(int(n) for n in selected.reshape(-1).tolist())
    del layer
    names = {
        f"model-moe-{STAGE:03d}-expert-{i:03d}.safetensors"
        for i in ids
    }
    sizes = remote_sizes({common_name, shared_name} | names)
    if sum(sizes.values()) > MAX_TOTAL_BYTES:
        raise RuntimeError("Full sparse-layer download exceeds 512 MiB")
    downloaded = fetch_to_cache(names, args.cache_dir)
    weights = {
        i: load_file(str(downloaded[f"model-moe-{STAGE:03d}-expert-{i:03d}.safetensors"]), device="cpu")
        for i in sorted(ids)
    }
    stats = run_layer(
        config, common, shared, weights,
        physical=physical, seed=args.seed, device=device,
    )
    print(f"IQ_REAL_MOE_EXPERT_SHARDS_LOADED={len(weights)}", flush=True)
    print("IQ_REAL_MOE_INFERENCE_SCOPE=one_isolated_layer_not_full_model", flush=True)


if __name__ == "__main__":
    main()
