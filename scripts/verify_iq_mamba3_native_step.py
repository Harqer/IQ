"""Fail-fast H200 correctness probe of the PINNED upstream Mamba-3 MIMO decode.

Uses the REAL CUDA fused prefill and CuTe/CUTLASS-DSL recurrent step on the
same GPU before attempting the large IQ checkpoint. No SISO/CPU emulation,
remote shard fetch, mock kernel, or new Hugging Face Job.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch


@torch.inference_mode()
def verify_native_mimo_step(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real Mamba-3 MIMO verification requires CUDA")
    from mamba_ssm import Mamba3

    model = Mamba3(
        d_model=256, d_state=128, headdim=64, expand=2,
        is_mimo=True, mimo_rank=4, chunk_size=16,
        is_outproj_norm=False, layer_idx=0, dtype=torch.bfloat16,
        device=device,
    ).eval()
    inputs = torch.randn(
        (1, 17, 256), device=device, dtype=torch.bfloat16,
    ) * 0.1
    state = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})
    # Real MIMO prefill must return all four recurrent state components.
    model(inputs[:, :16], inference_params=state)
    values = state.key_value_memory_dict.get(0)
    if values is None or len(values) != 4:
        raise RuntimeError("Native Mamba-3 MIMO did not return four prefill states")
    recurrent, *_ = model.step(inputs[:, 16], *values)
    reference = model(inputs)[:, 16, :]
    if recurrent.shape != reference.shape:
        raise RuntimeError("Native MIMO step has wrong output shape")
    if not bool(torch.isfinite(recurrent).all()):
        raise RuntimeError("Native MIMO step returned non-finite values")
    error = (recurrent.float() - reference.float()).abs()
    max_absolute = float(error.max())
    # BF16 fused and CuTe decode have different reduction orders. Require
    # a reasonably strict mixed absolute/relative tolerance, not exact bits.
    torch.testing.assert_close(
        recurrent.float(), reference.float(), atol=0.05, rtol=0.05,
    )
    result = {"max_absolute_error": max_absolute, "atol": 0.05, "rtol": 0.05}
    print(f"IQ_NATIVE_MIMO_PREFILL_DECODE_PARITY_PASS={result}", flush=True)
    del model, inputs, state
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def verify_full78_mimo_geometry(device: torch.device, config) -> dict[str, object]:
    """Fail closed on IQ's *actual* MIMO width/heads and multi-layer prefill.

    Original smoke probes use d_model=256; IQ full78 uses d_model=4096.
    This checks the exact checkpoint geometry at two independent native MIMO
    layers, including real fused return-state prefill, tail chunks and CuTe
    decoding. It runs before the 36-GiB base checkpoint is loaded.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full78 geometry requires real CUDA")
    from mamba_ssm import Mamba3

    options = dict(
        d_model=config.d_model,
        d_state=config.d_state,
        expand=config.expand,
        headdim=config.headdim,
        rope_fraction=config.rope_fraction,
        is_outproj_norm=config.outproj_norm,
        is_mimo=True,
        mimo_rank=config.mimo_rank,
        chunk_size=config.chunk_size,
        n_layer=config.num_layers,
        dtype=torch.bfloat16,
        device=device,
    )
    if config.d_model <= 256 or config.mimo_rank != 4:
        raise ValueError("Expected full78 native width >256 and MIMO rank 4")
    if config.d_model * config.expand % config.headdim != 0:
        raise ValueError("MIMO head geometry is inconsistent")
    print(
        "IQ_NATIVE_FULL78_GEOMETRY_START="
        f"width={config.d_model} state={config.d_state} "
        f"heads={int(config.d_model*config.expand/config.headdim)} "
        f"headdim={config.headdim} rank={config.mimo_rank} "
        f"chunk={config.chunk_size}", flush=True,
    )
    previous = Mamba3(**options, layer_idx=0).eval()
    current = Mamba3(**options, layer_idx=1).eval()
    # Two sizes cover an incomplete final chunk and a chunk boundary. The
    # second layer receives outputs from a preceding, equally wide mixer.
    max_abs = 0.0
    try:
        for length in (17, 33):
            source = torch.randn(
                (1, length, config.d_model),
                device=device, dtype=torch.bfloat16,
            ) * 0.01
            first = previous(source)
            torch.cuda.synchronize(device)
            print(f"IQ_NATIVE_FULL78_FIRST_LAYER_PASS=tokens={length}", flush=True)
            state = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})
            second_prefix = current(first[:, :-1], inference_params=state)
            torch.cuda.synchronize(device)
            values = state.key_value_memory_dict.get(1)
            if values is None or len(values) != 4:
                raise RuntimeError("Full78 MIMO geometry state missing")
            step, *_ = current.step(first[:, -1], *values)
            torch.cuda.synchronize(device)
            full = current(first)[:, -1]
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                step.float(), full.float(), atol=0.05, rtol=0.05,
            )
            if second_prefix.shape != first[:, :-1].shape:
                raise RuntimeError("Full78 geometry prefill changed width")
            error = float((step.float() - full.float()).abs().max())
            max_abs = max(max_abs, error)
            print(
                f"IQ_NATIVE_FULL78_GEOMETRY_PASS=tokens={length} "
                f"max_abs={error:.6f}", flush=True,
            )
    finally:
        del previous, current
        torch.cuda.empty_cache()
    return {"width": config.d_model, "checked_lengths": [17, 33],
            "max_absolute_error": max_abs}


if __name__ == "__main__":
    verify_native_mimo_step(torch.device("cuda:0"))
