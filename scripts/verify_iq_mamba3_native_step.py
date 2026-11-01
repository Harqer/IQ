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


if __name__ == "__main__":
    verify_native_mimo_step(torch.device("cuda:0"))
