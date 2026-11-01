#!/usr/bin/env python3
"""Full78 IQ inference smoke with a single H200 and ON-DEMAND routed experts.

Execute all 235 layers (Mamba-3 MIMO + CSA/HCA + Stable LatentMoE) and native
AttnRes/reasoning using genuine saved BF16 weights. No model weights are altered.
Mount a full78 HF bucket READ ONLY at /checkpoints and a verified Llama 3.1
tokenizer at /tokenizer. This script does not create Jobs or manage billing.

All *non-routed-expert* weights are loaded strictly. For each MoE forward, its
real router selects up to 16 distinct experts, which are then read individually
from the immutable safetensors bucket, executed and unloaded. The full 502 GiB
model remains logically complete, but the 19,200 experts are never all resident.

This is experimental first-token inference: failed kernels, I/O, or finite
checks fail closed. No quantization, SISO Mamba fallback or zeroed experts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any

import torch
from torch import nn
from safetensors.torch import load_file
from scripts.iq_h200_staging import iter_staged_shards
import os

from iq_model.architecture import HybridLayerType
from iq_model.hybrid import IQHybridConfig, IQHybridForCausalLM, UnloadedRoutedExpert
from iq_model.mlp.latent_moe import SiTUExpert
from scripts.preflight_glm53_full78_hf_inference import (
    EXPECTED_SHARDS, EXPECTED_TENSORS, validate_full78,
)

RUN = "glm53-full78-local-20261025a"
REQUIRED_DEVICE_MEMORY_GIB = 120
MAX_INPUT_TOKENS = 2
MAX_ACTIVE_EXPERTS = 16
MAX_BASELINE_GIB = 90
REPORT_PREFIX = "IQ_HF_FULL78"
SPECIAL_IDS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009,
}


def read_checkpoint(root: Path) -> tuple[dict, IQHybridConfig, dict[str, str]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    cfg = json.loads((root / "iq_config.json").read_text())
    complete = json.loads((root / "COMPILE_COMPLETE.json").read_text())
    report = validate_full78(index, cfg, complete)
    config = IQHybridConfig.from_dict(cfg)
    if config.moe_variant != "stable_latent" or config.stable_moe is None:
        raise RuntimeError("Full78 sparse inference requires stable_latent MoE")
    if config.multimodal is not None:
        raise RuntimeError("Multimodal full78 is not part of this text-only smoke")
    print(f"{REPORT_PREFIX}_INDEX_PASS=" + json.dumps(report, sort_keys=True), flush=True)
    return report, config, index["weight_map"]


def partition_weight_map(weight_map: dict[str, str]) -> tuple[dict[str, list[str]], dict[int, dict[int, str]]]:
    """Reject incomplete/ambiguous routed experts; return base and exact expert files."""
    base: dict[str, list[str]] = defaultdict(list)
    expert: dict[int, dict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for key, filename in weight_map.items():
        marker = ".moe.routed_experts."
        if marker not in key:
            base[filename].append(key)
            continue
        prefix, suffix = key.split(marker, 1)
        if not prefix.startswith("layers.") or suffix.count(".") != 2:
            raise RuntimeError(f"Unsafe routed expert mapping {key}")
        idx = int(prefix.split(".")[1])
        slot, tensor = suffix.split(".", 1)
        expert[idx][int(slot)][tensor] = filename
    routed: dict[int, dict[int, str]] = {}
    for idx, slots in expert.items():
        if set(slots) != set(range(256)):
            raise RuntimeError(f"Incomplete routing inventory in physical layer {idx}")
        routed[idx] = {}
        for slot, tensors in slots.items():
            if set(tensors) != {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}:
                raise RuntimeError(f"Incomplete GLM routed expert {idx}:{slot}")
            names = set(tensors.values())
            if len(names) != 1:
                raise RuntimeError(f"Expert tensors unexpectedly split across files: {idx}:{slot}")
            routed[idx][slot] = next(iter(names))
    if len(routed) != 78 or sum(map(len, routed.values())) != 78 * 256:
        raise RuntimeError("Wrong full78 19,968 expert slots (78 x 256)")
    if len(base) + sum(len(v) for v in routed.values()) != EXPECTED_SHARDS:
        raise RuntimeError("Shard-map inventory includes duplicate/misplaced files")
    return dict(base), routed


@torch.inference_mode()
def validate_and_copy_base_tensors(
    tensors: dict[str, torch.Tensor],
    expected_keys: set[str],
    params: dict[str, nn.Parameter],
    buffers: dict[str, torch.Tensor],
) -> tuple[float, float]:
    """Copy real BF16 checkpoint values and verify finiteness on destination.

    A per-tensor CPU BF16 `isfinite(...).all().item()` serialized every
    shard and dominated H200 startup. Copies now precede device-side finite
    reductions, with one synchronization per shard. Fail closed on NaN/Inf
    after the final GPU transfer; inference never sees invalid weights.
    """
    if set(tensors) != expected_keys:
        raise RuntimeError(
            "Base checkpoint shard keys differ: "
            f"missing={sorted(expected_keys-set(tensors))[:10]} "
            f"extra={sorted(set(tensors)-expected_keys)[:10]}"
        )
    destinations: list[tuple[str, torch.Tensor]] = []
    for key, value in tensors.items():
        target = params.get(key)
        if target is None:
            target = buffers.get(key)
        if target is None or tuple(value.shape) != tuple(target.shape):
            raise RuntimeError(f"Base checkpoint tensor shape mismatch: {key}")
        if not value.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise RuntimeError(f"Nonfloating checkpoint tensor: {key}")
        destinations.append((key, target))
    if not destinations:
        raise RuntimeError("Empty baseline checkpoint shard")
    device = destinations[0][1].device
    if any(target.device != device for _, target in destinations):
        raise RuntimeError("Baseline tensors span multiple devices unexpectedly")
    copy_started = perf_counter()
    for key, target in destinations:
        # copy_ converts dtype as required, avoiding a second temporary
        # CUDA allocation from value.to(...). Read-only source is unchanged.
        target.copy_(tensors[key], non_blocking=(device.type == "cuda"))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    copy_seconds = perf_counter() - copy_started

    finite_started = perf_counter()
    finite_checks = [torch.isfinite(target).all() for _, target in destinations]
    if not bool(torch.stack(finite_checks).all().item()):
        invalid = [
            key for key, target in destinations
            if not bool(torch.isfinite(target).all().item())
        ]
        raise RuntimeError(f"Nonfinite checkpoint tensor(s) after copy: {invalid[:10]}")
    finite_seconds = perf_counter() - finite_started
    return copy_seconds, finite_seconds


@torch.inference_mode()
def load_base_weights(model: IQHybridForCausalLM, root: Path,
                      base: dict[str, list[str]], *,
                      budget_gib: int = MAX_BASELINE_GIB) -> None:
    expected = set(model.state_dict())
    actual = {key for keys in base.values() for key in keys}
    if expected != actual:
        raise RuntimeError(
            f"Non-expert checkpoint/state mismatch: "
            f"missing={sorted(expected - actual)[:10]} "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    total_size = 0
    for name in sorted(base):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing/symlinked read-only checkpoint shard {name}")
        total_size += path.stat().st_size
    if total_size > budget_gib * 1024**3:
        raise RuntimeError(f"Base tensors exceed bounded GPU residency {budget_gib} GiB")
    print(f"{REPORT_PREFIX}_BASELINE_BYTES={total_size} shards={len(base)}", flush=True)
    workers_raw = os.getenv("IQ_BASELINE_STAGE_WORKERS", "4")
    try:
        workers = int(workers_raw)
    except ValueError as exc:
        raise RuntimeError("IQ_BASELINE_STAGE_WORKERS must be an integer 1..8") from exc
    stage_parent_raw = os.getenv("IQ_BASELINE_STAGE_DIR", "")
    stage_parent = Path(stage_parent_raw) if stage_parent_raw else None
    print(
        f"{REPORT_PREFIX}_BASE_STAGE_CONFIG=workers={workers} "
        f"dir={stage_parent or 'system-temp'}",
        flush=True,
    )
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    baseline_started = perf_counter()
    loaded_bytes = 0
    loaded_count = 0
    # This iterator overlaps bounded remote bucket reads with CPU validation
    # and CUDA copies. Each staged file is removed immediately after use.
    for i, (name, shard_path, staged_bytes, stage_seconds) in enumerate(
        iter_staged_shards(root, sorted(base), workers=workers, stage_parent=stage_parent), 1
    ):
        expected_keys = set(base[name])
        started = perf_counter()
        print(
            f"{REPORT_PREFIX}_BASE_SHARD_STAGED={i}/{len(base)} "
            f"file={name} bytes={staged_bytes} "
            f"stage_seconds={stage_seconds:.2f}",
            flush=True,
        )
        decoded_at = perf_counter()
        tensors = load_file(str(shard_path), device="cpu")
        decode_seconds = perf_counter() - decoded_at
        copy_seconds, finite_seconds = validate_and_copy_base_tensors(
            tensors, expected_keys, params, buffers,
        )
        del tensors
        loaded_bytes += staged_bytes
        loaded_count = i
        print(
            f"{REPORT_PREFIX}_BASE_SHARD_LOADED={i}/{len(base)} "
            f"file={name} decode_seconds={decode_seconds:.2f} "
            f"gpu_copy_seconds={copy_seconds:.2f} "
            f"finite_check_seconds={finite_seconds:.2f} "
            f"total_seconds={perf_counter()-started:.2f}",
            flush=True,
        )
        if i % 25 == 0 or i == len(base):
            free, _ = torch.cuda.mem_get_info()
            print(
                f"{REPORT_PREFIX}_BASE_LOADED={i}/{len(base)} "
                f"cuda_free_gib={free/1024**3:.1f} "
                f"elapsed_seconds={perf_counter()-baseline_started:.2f}",
                flush=True,
            )
    if loaded_count != len(base) or loaded_bytes != total_size:
        raise RuntimeError("Incomplete staged baseline load")
    print(
        f"{REPORT_PREFIX}_BASELINE_LOAD_PASS=shards={loaded_count} "
        f"bytes={loaded_bytes} "
        f"seconds={perf_counter()-baseline_started:.2f}",
        flush=True,
    )

def hydrate_expert(shard: Path, *, prefix: str, hidden: int,
                   intermediate: int, beta: float, linear_beta: float,
                   device: torch.device, dtype: torch.dtype,
                   bias: bool) -> SiTUExpert:
    with torch.device("meta"):
        expert = SiTUExpert(
            hidden, intermediate, beta=beta, linear_beta=linear_beta, bias=bias,
        )
    # Stored GLM experts have exactly three weights and no bias.
    if bias:
        raise RuntimeError("Unsupported MoE expert-bias geometry")
    saved = load_file(str(shard), device="cpu")
    expected = {prefix + kind + "_proj.weight" for kind in ("gate", "up", "down")}
    if set(saved) != expected:
        raise RuntimeError(f"Selected expert shard index mismatch {shard.name}")
    for full_key, src in saved.items():
        name = full_key[len(prefix):]
        which, _, param_name = name.partition(".")
        target = getattr(expert, which)
        if param_name != "weight" or tuple(target.weight.shape) != tuple(src.shape):
            raise RuntimeError(f"Selected expert geometry mismatch {full_key}")
        if not bool(torch.isfinite(src).all()):
            raise RuntimeError(f"Nonfinite selected expert weight {full_key}")
        target.weight = nn.Parameter(
            src.to(device=device, dtype=dtype), requires_grad=False,
        )
    return expert.eval()


class ExpertStreamer:
    """Pre-hook hydrates routed weights; post-hook releases them each forward."""

    def __init__(self, root: Path, model: IQHybridForCausalLM,
                 expert_map: dict[int, dict[int, str]], *,
                 max_active: int = MAX_ACTIVE_EXPERTS):
        self.root = root
        self.model = model
        self.expert_map = expert_map
        self.max_active = max_active
        self.forward_count = 0
        self.experts_loaded = 0
        self.handles = []
        for physical, layer in enumerate(model.layers):
            if physical not in expert_map:
                continue
            self.handles.append(layer.register_forward_pre_hook(
                lambda mod, inputs, n=physical: self.before(n, mod, inputs),
            ))
            self.handles.append(layer.register_forward_hook(
                lambda mod, inputs, output, n=physical: self.after(n, mod, output),
            ))

    def before(self, physical: int, layer: nn.Module, inputs: tuple) -> None:
        if not inputs:
            raise RuntimeError("MoE loader got empty layer inputs")
        moe = layer.moe
        x = layer.norm(inputs[0])
        route = moe._route(x.reshape(-1, x.shape[-1]))[1]
        ids = sorted(set(int(i) for i in route.reshape(-1).tolist()))
        if not ids or len(ids) > self.max_active:
            raise RuntimeError(
                f"Too many live experts for bounded smoke: {physical} -> {len(ids)}"
            )
        conf = moe.config
        device = x.device
        for slot in ids:
            if not isinstance(moe.routed_experts[slot], UnloadedRoutedExpert):
                raise RuntimeError("Stale MoE expert occupied a slot")
            path = self.root / self.expert_map[physical][slot]
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Cannot read selected HF bucket expert {physical}:{slot}")
            prefix = f"layers.{physical}.moe.routed_experts.{slot}."
            moe.routed_experts[slot] = hydrate_expert(
                path, prefix=prefix, hidden=conf.latent_size,
                intermediate=conf.expert_intermediate_size,
                beta=conf.situ_beta, linear_beta=conf.situ_linear_beta,
                device=device, dtype=x.dtype, bias=conf.expert_bias,
            )
        # The actual production MoE forward repeats exact routing with
        # unchanged layer input; it cannot execute any unselected Identity.
        layer._iq_live_experts = ids

    def after(self, physical: int, layer: nn.Module, output: Any) -> None:
        ids = getattr(layer, "_iq_live_experts", None)
        if ids is None:
            raise RuntimeError(f"MoE post-hook without hydrated expert set at {physical}")
        observed = set(int(i) for i in output.selected_experts.reshape(-1).tolist())
        if observed != set(ids):
            raise RuntimeError(
                f"Actual MoE execution selected unexpected experts at {physical}: "
                f"loaded={ids} observed={sorted(observed)}"
            )
        for slot in ids:
            layer.moe.routed_experts[slot] = UnloadedRoutedExpert()
        del layer._iq_live_experts
        self.forward_count += 1
        self.experts_loaded += len(ids)
        if self.forward_count % 10 == 0 or self.forward_count == 78:
            print(
                f"{REPORT_PREFIX}_MOE_PROGRESS={self.forward_count}/78 "
                f"selected_shards={self.experts_loaded}", flush=True,
            )

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def validate_tokenizer(directory: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    if len(tokenizer) != 128256:
        raise RuntimeError("HF tokenizer vocabulary must be exactly Llama 3.1 / 128256")
    for token, idx in SPECIAL_IDS.items():
        if tokenizer.convert_tokens_to_ids(token) != idx:
            raise RuntimeError(f"HF tokenizer special-token mismatch: {token}")
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/checkpoints") / "IQ" / "runs" / RUN)
    parser.add_argument("--tokenizer", type=Path, default=Path("/tokenizer"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("This is CUDA Mamba-3 MIMO inference, not a CPU/fake kernel test")
    dev = torch.device(f"cuda:{args.cuda_device}")
    mem = torch.cuda.get_device_properties(dev).total_memory
    if mem < REQUIRED_DEVICE_MEMORY_GIB * 1024**3:
        parser.error("Require one >=120 GiB GPU for this experimental single-device sparse smoke")
    report, config, weight_map = read_checkpoint(args.checkpoint)
    base, expert_map = partition_weight_map(weight_map)
    print(f"{REPORT_PREFIX}_BASE_SHARDS={len(base)} routed_shards={78*256}", flush=True)

    tokenizer = validate_tokenizer(args.tokenizer)
    ids = [SPECIAL_IDS["<|begin_of_text|>"]] + tokenizer.encode(
        args.prompt, add_special_tokens=False,
    )
    if not 1 <= len(ids) <= MAX_INPUT_TOKENS:
        parser.error(f"Use a 1- or 2-token prompt; got {len(ids)} tokens")
    print(f"{REPORT_PREFIX}_PROMPT_TOKEN_IDS={ids}", flush=True)

    # Use exact Mamba-3 MIMO and full 235-layer IQ forward, never SISO/CPU.
    model = IQHybridForCausalLM(
        config, dtype=torch.bfloat16, device=dev, sparse_routed_moe=True,
    ).eval()
    load_base_weights(model, args.checkpoint, base)
    streamer = ExpertStreamer(args.checkpoint, model, expert_map)
    try:
        with torch.inference_mode():
            result = model(torch.tensor([ids], device=dev, dtype=torch.long))
        if streamer.forward_count != 78:
            raise RuntimeError(f"Incomplete full78 routing: {streamer.forward_count}/78")
        logits = result.logits
        if logits.shape != (1, len(ids), 128256):
            raise RuntimeError(f"Full IQ logits geometry mismatch {tuple(logits.shape)}")
        last = logits[0, -1].float()
        if not bool(torch.isfinite(last).all()):
            raise RuntimeError("Full78 inference output contains NaN/Inf logits")
        probabilities = torch.softmax(last, dim=-1)
        top = torch.topk(probabilities, k=5)
        selected = int(top.indices[0].item())
        if not 0 <= selected < 128256:
            raise RuntimeError("Out-of-vocab IQ token")
        print(f"{REPORT_PREFIX}_FIRST_TOKEN_PASS=" + json.dumps({
            "full_physical_layers": 235,
            "mamba_layers": 79,
            "moe_layers_executed": streamer.forward_count,
            "selected_expert_shards_loaded": streamer.experts_loaded,
            "prompt_ids": ids,
            "predicted_token_id": selected,
            "predicted_text": tokenizer.decode([selected]),
            "top5_ids": top.indices.tolist(),
            "top5_probs": top.values.tolist(),
            "checkpoint_weights": "full78 BF16; sparse routed experts loaded on demand",
        }, sort_keys=True), flush=True)
    finally:
        streamer.close()
    print(f"{REPORT_PREFIX}_SMOKE_SCOPE=one_first_token_not_knowledge_retention", flush=True)


if __name__ == "__main__":
    main()
