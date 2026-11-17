#!/usr/bin/env python3
"""Full-geometry IQ hybrid execution on a Colab GPU, with RANDOM weights.

This is a *single real IQHybridForCausalLM.forward call*: all eighty layers,
all 32 experts in each of the 24 Stable LatentMoEs, CSA/HCA, AttnRes and
reasoning are instantiated at the *unchanged* canonical dimensions.

Weights are initialized only as the actual layer executes and released when
its output is computed, using meta modules + forward hooks. Nothing is
loaded from a checkpoint or external model.

IMPORTANT: Mamba-3 MIMO is executed by the rank-4 pure-Torch mathematical
oracle, not by the production TileLang/CUTLASS GPU kernel. FP16 is used on
a T4 (BF16 may be selected on supported devices). This verifies full
production GEOMETRY and cross-module EXECUTION, not production kernel parity,
throughput, pretrained reasoning, or full resident weight allocation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from iq_model import IQHybridForCausalLM
from iq_model.architecture import HybridLayerType
from iq_transfer.complete_transplant import canonical_complete_config

EXPECTED_SCHEDULE = {
    HybridLayerType.MAMBA3: 32,
    HybridLayerType.CSA: 12,
    HybridLayerType.HCA: 12,
    HybridLayerType.MOE: 24,
}
REPORT_PATH = Path(os.getenv("IQ_FULL_GEOMETRY_REPORT", "/content/iq_full_random_report.json"))


def write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(REPORT_PATH)


@torch.no_grad()
def initialize_random_module(module: nn.Module) -> None:
    """Initialize every materialized tensor; never read uninitialized memory.

    Linear and embedding modules use PyTorch's normal reset_parameters.
    For recurrent and custom modules, preserve structural positivity/stability
    by initializing time parameters, norms and MIMO projections intentionally.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Linear, nn.Embedding)):
            sub.reset_parameters()
            continue
        for name, parameter in sub.named_parameters(recurse=False):
            if name == "bias" or name == "router_bias":
                parameter.zero_()
            elif name == "mix_logit":
                parameter.fill_(-12)
            elif name == "dt_bias":
                parameter.fill_(-3)
            elif name in ("B_bias", "C_bias", "D"):
                parameter.fill_(1)
            elif name in ("mimo_x", "mimo_o"):
                parameter.fill_(1.0 / parameter.shape[1])
            elif name == "mimo_z":
                parameter.fill_(1)
            elif name == "weight" and parameter.ndim == 1:
                parameter.fill_(1)
            elif parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)
            else:
                parameter.zero_()
        for _, buffer in sub.named_buffers(recurse=False):
            buffer.zero_()


def materialize(module: nn.Module, device: torch.device) -> None:
    if any(not p.is_meta for p in module.parameters()):
        raise RuntimeError("Attempted to materialize an already resident module")
    module.to_empty(device=device)
    initialize_random_module(module)
    module.eval()


def release(module: nn.Module) -> None:
    module.to_empty(device=torch.device("meta"))


def tensor_is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def assert_canonical(cfg) -> None:
    if cfg.model.hidden_size != 4096 or cfg.model.vocab_size != 201088:
        raise AssertionError("Production hidden width or vocabulary changed")
    if cfg.mamba3.mimo_rank != 4 or cfg.mamba3.d_state != 128 or cfg.mamba3.headdim != 64:
        raise AssertionError("MIMO production geometry changed")
    if cfg.stable_moe is None or cfg.stable_moe.latent_size != 2880:
        raise AssertionError("Stable LatentMoE geometry changed")
    if cfg.stable_moe.num_experts != 32 or cfg.stable_moe.top_k != 4:
        raise AssertionError("MoE expert geometry changed")
    if cfg.moe_variant != "stable_latent":
        raise AssertionError("Stable LatentMoE was disabled")
    if cfg.attnres is None or cfg.reasoning is None or cfg.compressed_context is None:
        raise AssertionError("Required architecture modules were disabled")
    if len(cfg.schedule.layers) != 80:
        raise AssertionError("Production layer schedule is not 80 layers")
    for kind, expected in EXPECTED_SCHEDULE.items():
        if cfg.schedule.count(kind) != expected:
            raise AssertionError(f"{kind.value} schedule differs from canonical: {expected}")


def run_geometry(*, tokens: int = 4, compression_probe: bool = True) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No accessible Colab CUDA GPU. Select Runtime > Change runtime type "
            "> GPU; this program will NOT claim a CPU-only full test passed."
        )
    gpu = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cfg = canonical_complete_config()
    assert_canonical(cfg)
    if not 1 <= tokens <= 16:
        raise ValueError("Use 1..16 tokens for first full-width execution, not scaled layers")
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.cuda.reset_peak_memory_stats(gpu)
    gpu_name = torch.cuda.get_device_name(gpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
    print(f"IQ production-geometry test on {gpu_name}, {total_bytes/2**30:.2f} GiB VRAM")
    print(f"Native dtype: {dtype}. No checkpoint. Exact 80-layer schedule.")
    if total_bytes < 8 * 2**30:
        raise RuntimeError("At least 8 GiB VRAM required for layer-streamed full-width tensors")

    report = {
        "status": "in_progress",
        "mode": "full_geometry_layer_streamed_random_weights",
        "full_forward_method": "IQHybridForCausalLM.forward",
        "mamba_engine": "rank4_pure_torch_reference_not_production_tilelang",
        "gpu": gpu_name,
        "gpu_total_gib": round(total_bytes / 2**30, 3),
        "gpu_free_gib_start": round(free_bytes / 2**30, 3),
        "dtype": str(dtype),
        "seed": 1729,
        "pretrained_weights_loaded": False,
        "config_fingerprint": cfg.fingerprint,
        "hidden_size": cfg.model.hidden_size,
        "vocab_size": cfg.model.vocab_size,
        "layers": len(cfg.schedule.layers),
        "schedule": {k.value: v for k, v in EXPECTED_SCHEDULE.items()},
        "moe_experts_per_layer": cfg.stable_moe.num_experts,
        "moe_top_k": cfg.stable_moe.top_k,
        "csa_compress_rate": cfg.compressed_context.csa_compress_rate,
        "hca_compress_rate": cfg.compressed_context.hca_compress_rate,
        "tokens": tokens,
        "completed_layer_indices": [],
        "component_counts": {},
        "moe_routed_token_counts": [],
        "compression_probes": {},
    }
    write_report(report)

    # Model weights occupy META storage until the exact module is called.
    # Mamba is a full-width rank-4 Torch reference (no SISO substitution).
    with torch.device("meta"):
        model = IQHybridForCausalLM(
            cfg, dtype=dtype, device="meta", test_only_cpu_reference=True
        )
    if len(model.layers) != 80:
        raise AssertionError("Wrong number of instantiated physical layers")
    for layer_idx, kind in enumerate(cfg.schedule.layers):
        layer = model.layers[layer_idx]
        if kind is HybridLayerType.MOE:
            if len(layer.moe.routed_experts) != 32:
                raise AssertionError(f"layer {layer_idx}: not 32 real routed experts")
            if len(layer.moe.shared_experts) != cfg.stable_moe.num_shared_experts:
                raise AssertionError(f"layer {layer_idx}: shared experts absent")
        if kind in (HybridLayerType.CSA, HybridLayerType.HCA):
            expected_mode = "csa" if kind is HybridLayerType.CSA else "hca"
            if layer.attention.mode != expected_mode:
                raise AssertionError(f"layer {layer_idx}: wrong attention mode")

    callbacks = []
    layer_t0 = {}
    count = Counter()
    # Non-__call__ state machinery; only a few small parameters persist.
    for persist in (model.attnres, model.norm):
        if persist is not None:
            materialize(persist, gpu)

    def add_streaming_hooks(module: nn.Module, *, label: str, index: int | None):
        def before(own, _args):
            layer_t0[label] = time.monotonic()
            materialize(own, gpu)
        def after(own, _args, result):
            torch.cuda.synchronize(gpu)
            value = getattr(result, "hidden_states", None)
            if value is None:
                value = getattr(result, "state", None)
            if value is None:
                value = result
            if not isinstance(value, torch.Tensor) or not tensor_is_finite(value):
                raise FloatingPointError(f"{label} returned non-finite/invalid output")
            if index is not None:
                kind = cfg.schedule.layers[index]
                if kind is HybridLayerType.MOE:
                    counts_tensor = result.expert_counts
                    total_assigned = int(counts_tensor.sum().item())
                    wanted = tokens * cfg.stable_moe.top_k
                    if total_assigned != wanted or counts_tensor.numel() != 32:
                        raise AssertionError(f"MoE layer {index}: route counts incorrect")
                    report["moe_routed_token_counts"].append(total_assigned)
                count[kind.value] += 1
                report["completed_layer_indices"].append(index)
                report["component_counts"] = dict(count)
            duration = round(time.monotonic() - layer_t0[label], 3)
            release(own)
            torch.cuda.empty_cache()
            write_report(report)
            print(
                f"{label} done ({duration}s; "
                f"peak VRAM {torch.cuda.max_memory_allocated(gpu)/2**30:.2f} GiB)",
                flush=True,
            )
        callbacks.append(module.register_forward_pre_hook(before))
        callbacks.append(module.register_forward_hook(after))

    for i, layer in enumerate(model.layers):
        kind = cfg.schedule.layers[i]
        add_streaming_hooks(layer, label=f"{i+1:02d}/80:{kind.value}", index=i)
    add_streaming_hooks(model.embed_tokens, label="embedding", index=None)
    add_streaming_hooks(model.lm_head, label="lm_head", index=None)
    if model.reasoning is not None:
        add_streaming_hooks(model.reasoning, label="reasoning", index=None)
    if model.reasoning_injector is not None:
        add_streaming_hooks(model.reasoning_injector, label="reasoning_injector", index=None)

    # Real embedding 201088 x 4096 and real LM head dimensions are executed,
    # but not resident at the same time as any MoE layer.
    token_ids = torch.arange(100, 100 + tokens, dtype=torch.long, device=gpu).unsqueeze(0)
    start = time.monotonic()
    try:
        with torch.inference_mode():
            result = model(input_ids=token_ids, return_hidden_states=True)
            torch.cuda.synchronize(gpu)
            if result.logits.shape != (1, tokens, cfg.model.vocab_size):
                raise AssertionError("Full-width output head returned unexpected logits")
            if not tensor_is_finite(result.logits):
                raise FloatingPointError("Non-finite logits")
            if len(report["completed_layer_indices"]) != 80:
                raise AssertionError("Not all 80 production layers executed")
            if count != Counter({k.value:v for k,v in EXPECTED_SCHEDULE.items()}):
                raise AssertionError("Missing or repeated architecture components")
            if len(result.expert_counts) != 24:
                raise AssertionError("Missing mandatory Stable LatentMoE outputs")
            if result.reasoning_state is None:
                raise AssertionError("Reasoning recurrence was not executed")
            report["output_shape"] = list(result.logits.shape)
            report["logits_checksum"] = float(result.logits.float().sum().item())
            report["finite_logits"] = True

            # Independent full-width CSA and HCA probe at the shortest
            # lengths that activate complete compression groups. The
            # end-to-end test above uses small token count because running
            # 129 x 80 full-width reference layers is prohibitively costly.
            if compression_probe:
                from iq_model.hybrid import CompressedContextResidualLayer
                for kind, length in (
                    (HybridLayerType.CSA, cfg.compressed_context.csa_compress_rate * 2 + 1),
                    (HybridLayerType.HCA, cfg.compressed_context.hca_compress_rate + 1),
                ):
                    probe = None
                    x = y = None
                    label = kind.value
                    try:
                        with torch.device("meta"):
                            probe = CompressedContextResidualLayer(
                                cfg.model, cfg.compressed_context, mode=kind
                            )
                        materialize(probe, gpu)
                        x = torch.randn((1,length,4096),device=gpu,dtype=dtype) * 0.01
                        y = probe(x)
                        torch.cuda.synchronize(gpu)
                        if y.shape != x.shape or not tensor_is_finite(y):
                            raise AssertionError(f"{label} compression output invalid")
                        report["compression_probes"][label] = {
                            "tokens": length, "finite": True,
                            "shape": list(y.shape), "output_checksum":float(y.float().sum())
                        }
                        print(f"{label} real full-width compression probe passed (seq={length})")
                    finally:
                        del x, y
                        if probe is not None:
                            del probe
                        torch.cuda.empty_cache()
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["exception"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=12)
        raise
    finally:
        for callback in callbacks:
            callback.remove()
        report["elapsed_seconds"] = round(time.monotonic()-start,3)
        report["peak_vram_gib"] = round(
            torch.cuda.max_memory_allocated(gpu)/2**30,3
        )
        write_report(report)
    print(json.dumps({
        key:report[key] for key in (
            "status", "layers", "completed_layer_indices", "component_counts",
            "output_shape", "finite_logits", "compression_probes",
            "elapsed_seconds", "peak_vram_gib"
        )
    }, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument(
        "--no-compression-probes", action="store_true",
        help="Only full 80-layer execution; skip extra full-width compression probes",
    )
    args = parser.parse_args()
    run_geometry(
        tokens=args.tokens,
        compression_probe=not args.no_compression_probes,
    )
