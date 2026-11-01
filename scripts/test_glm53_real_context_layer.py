#!/usr/bin/env python3
"""Read-only real HCA checkpoint test on local CUDA: GLM stage 3, physical layer 10.

This loads one 131 MiB BF16 context shard, uses IQ's real
CompressedContextResidualLayer.forward, and checks causality, finite output
and attention contribution on a 144-token synthetic hidden-state input.
Does NOT run the preceding IQ layers or prove language generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file

from iq_model.architecture import HybridLayerType
from iq_model.attention.compressed import CompressedContextConfig
from iq_model.hybrid import CompressedContextResidualLayer

BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "glm53-full78-local-20261025a"
PREFIX = f"IQ/runs/{RUN}"
STAGE = 3
PHYSICAL = 10
SHARD = "model-context-003.safetensors"
MAX_BYTES = 200 * 1024**2


def get_remote_size() -> int:
    from huggingface_hub import get_bucket_paths_info
    path = f"{PREFIX}/{SHARD}"
    files = list(get_bucket_paths_info(BUCKET, [path]))
    if len(files) != 1 or files[0].path != path:
        raise RuntimeError(f"Missing HF bucket file: {path}")
    size = files[0].size
    if not isinstance(size, int) or not 1024 < size <= MAX_BYTES:
        raise RuntimeError(f"Unexpected HCA shard size: {size}")
    return size


def local_shard(cache_dir: Path) -> Path:
    """Download just one bounded shard; never upload/modify the HF checkpoint."""
    from huggingface_hub import download_bucket_files
    size = get_remote_size()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / SHARD
    if path.is_symlink():
        raise RuntimeError("Refusing symlinked local shard")
    if path.is_file():
        if path.stat().st_size != size:
            raise RuntimeError("Cached shard size mismatch; remove manually before retry")
        return path
    if path.exists():
        raise RuntimeError("Unexpected cached checkpoint path")
    with tempfile.TemporaryDirectory(prefix="iq-hca-", dir=cache_dir) as tmp:
        partial = Path(tmp) / SHARD
        download_bucket_files(
            BUCKET,
            files=[(f"{PREFIX}/{SHARD}", str(partial))],
            raise_on_missing_files=True,
        )
        if not partial.is_file() or partial.stat().st_size != size:
            raise RuntimeError("Incomplete HCA checkpoint download")
        os.replace(partial, path)
    print(f"IQ_REAL_HCA_SHARD_DOWNLOADED={SHARD} bytes={size}", flush=True)
    return path


def build_context_layer(config: dict, saved: dict[str, torch.Tensor],
                        *, physical: int = PHYSICAL) -> CompressedContextResidualLayer:
    """Exact state key/shape validation, no random or missing saved tensors."""
    if config["schedule"]["layers"][physical] != "hca":
        raise RuntimeError(f"Expected HCA at physical layer {physical}")
    model_cfg = config["model"]
    ccfg = CompressedContextConfig(**config["compressed_context"])
    if int(model_cfg["hidden_size"]) != ccfg.hidden_size:
        raise RuntimeError("Hidden-size mismatch in IQ context configuration")
    # No giant full-IQ model allocation. One layer on CPU, float32.
    layer = CompressedContextResidualLayer(
        _as_model_config(model_cfg), ccfg, mode=HybridLayerType.HCA,
    ).eval()
    prefix = f"layers.{physical}."
    expected = set(layer.state_dict())
    actual = {key.removeprefix(prefix) for key in saved if key.startswith(prefix)}
    if len(actual) != len(saved) or actual != expected:
        raise RuntimeError(
            f"Saved HCA tensor coverage mismatch: "
            f"missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}"
        )
    state = {}
    for full_name, value in saved.items():
        key = full_name[len(prefix):]
        current = layer.state_dict()[key]
        if (tuple(value.shape) != tuple(current.shape) or
            value.dtype not in (torch.bfloat16, torch.float32) or
            not bool(torch.isfinite(value).all())):
            raise RuntimeError(f"Saved HCA tensor invalid: {full_name}")
        state[key] = value.float()
    layer.load_state_dict(state, strict=True)
    return layer


def _as_model_config(data: dict):
    from iq_model.config import IQModelConfig
    return IQModelConfig(**data)


@torch.inference_mode()
def exercise_context(layer: CompressedContextResidualLayer, *,
                     device: torch.device, seed: int, sequence: int) -> dict:
    if sequence < layer.attention.config.hca_compress_rate + 4:
        raise RuntimeError("Require at least one real HCA compressed block")
    if sequence > 160:
        raise RuntimeError("Local HCA smoke must use at most 160 tokens")
    layer = layer.to(device=device, dtype=torch.float32).eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, sequence, layer.attention.config.hidden_size, generator=gen)
    x = x.to(device=device, dtype=torch.float32)
    out = layer(x, position_ids=None, attention_mask=None, document_ids=None)
    if out.shape != x.shape or not bool(torch.isfinite(out).all()):
        raise RuntimeError("Saved HCA forward produced invalid outputs")
    residual = float((out - x).float().norm().item())
    if not 0 < residual < float("inf"):
        raise RuntimeError("Saved HCA attention did not contribute to its residual")

    # Causal check: changing tokens in the future must not modify prior outputs.
    changed = x.clone()
    changed[:, -4:] += 10.
    other = layer(changed, position_ids=None, attention_mask=None, document_ids=None)
    prefix_delta = float((out[:, :-4] - other[:, :-4]).abs().max().item())
    if prefix_delta > 1e-5:
        raise RuntimeError(f"HCA causal isolation failed: max_prefix_change={prefix_delta}")

    # Explicitly remove the attended projection: output must revert to identity.
    final = layer.attention.output.out_proj
    original = final.weight.detach().clone()
    try:
        final.weight.zero_()
        ablated = layer(x, position_ids=None, attention_mask=None, document_ids=None)
    finally:
        final.weight.copy_(original)
    difference = float((out - ablated).abs().max().item())
    if difference <= 1e-6 or not difference < float("inf"):
        raise RuntimeError("Saved attention output projection does not affect output")
    stats = {
        "stage": STAGE, "physical_layer": PHYSICAL, "mode": "hca",
        "tokens": sequence, "hca_compress_rate": layer.attention.config.hca_compress_rate,
        "compute_dtype": "FP32", "checkpoint_dtype": "BF16",
        "device": str(device), "attention_residual_norm": residual,
        "causal_prefix_max_difference": prefix_delta,
        "output_projection_ablation_max_difference": difference,
    }
    print("IQ_REAL_CHECKPOINT_HCA_FORWARD_PASS=" + json.dumps(stats, sort_keys=True), flush=True)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/iq-real-context-stage003")
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--sequence", type=int, default=144)
    a = p.parse_args()
    if not os.environ.get("HF_TOKEN"):
        p.error('HF_TOKEN missing; export HF_TOKEN="$(hf auth token)"')
    if not torch.cuda.is_available():
        p.error("CUDA GPU required for actual local HCA forward")
    if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:
        p.error("Require >=4 GiB GPU memory for bounded HCA check")
    from huggingface_hub import get_bucket_paths_info
    paths = [f"{PREFIX}/iq_config.json", f"{PREFIX}/COMPILE_COMPLETE.json"]
    infos = {item.path: item.size for item in get_bucket_paths_info(BUCKET, paths)}
    if len(infos) != 2:
        raise RuntimeError("Missing checkpoint metadata")
    with tempfile.TemporaryDirectory(prefix="iq-real-hca-metadata-") as td:
        from huggingface_hub import download_bucket_files
        for name in ("iq_config.json", "COMPILE_COMPLETE.json"):
            download_bucket_files(
                BUCKET, files=[(f"{PREFIX}/{name}", str(Path(td) / name))],
                raise_on_missing_files=True,
            )
        config = json.loads((Path(td) / "iq_config.json").read_text())
        complete = json.loads((Path(td) / "COMPILE_COMPLETE.json").read_text())
    if (complete.get("status") != "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED" or
        config["schedule"]["layers"][PHYSICAL] != "hca" or
        len(config["schedule"]["layers"]) != 235):
        raise RuntimeError("Unexpected IQ full78 checkpoint version or schedule")
    file = local_shard(a.cache_dir)
    tensors = load_file(str(file), device="cpu")
    layer = build_context_layer(config, tensors)
    exercise_context(layer, device=torch.device("cuda:0"),
                     seed=a.seed, sequence=a.sequence)
    print("IQ_REAL_CONTEXT_SCOPE=one_isolated_HCA_layer_not_full_model", flush=True)


if __name__ == "__main__":
    main()
