"""Bounded, opt-in OTLP diagnostics for real IQ235 inference.

Only numerical aggregates, model geometry, and progress counts are exported.
Never emit prompts, token strings, token IDs, weights, credentials, or paths.
This module intentionally imports neither Torch nor OpenTelemetry at import.
"""
from __future__ import annotations

from contextlib import nullcontext


class IQ235NumericError(RuntimeError):
    """The first non-finite layer or invalid output during full inference."""


def span_scope(tracer, name: str, attributes: dict | None = None):
    return (tracer.start_as_current_span(name, attributes=attributes or {})
            if tracer is not None else nullcontext())


def emit_progress(span, done: int, total: int, loaded: int) -> None:
    """Avoid 20k spans/events; emit a compact event every 250 shards."""
    if span is not None and (done == 1 or done % 250 == 0 or done == total):
        span.add_event("iq.checkpoint.progress", {
            "iq.shards.verified": done,
            "iq.shards.expected": total,
            "iq.tensors.loaded": loaded,
        })


def examine_layer_output(torch, output, *, index: int, layer_type: str, span=None):
    """Check whole layer output only when IQ235_NUMERIC_DIAGNOSTICS=1.

    Numeric reductions synchronize CUDA and are intentionally opt-in.
    A failure includes layer index/type but never a raw activation.
    """
    value = output if torch.is_tensor(output) else getattr(output, "hidden_states", None)
    if not torch.is_tensor(value) or value.numel() == 0:
        raise IQ235NumericError(f"invalid tensor output at layer {index} ({layer_type})")
    detached = value.detach()
    finite = bool(torch.isfinite(detached).all().item())
    if not finite:
        if span is not None:
            span.add_event("iq.layer.nonfinite", {
                "iq.layer.index": index, "iq.layer.type": layer_type,
            })
        raise IQ235NumericError(f"nonfinite activation at layer {index} ({layer_type})")
    numeric = detached.float()
    maximum = float(numeric.abs().amax().item())
    rms = float(numeric.square().mean().sqrt().item())
    if span is not None:
        span.add_event("iq.layer.numeric", {
            "iq.layer.index": index,
            "iq.layer.type": layer_type,
            "iq.layer.abs_max": maximum,
            "iq.layer.rms": rms,
        })
    return maximum, rms


def write_experimental_result(
    manifest_path: str, *, status: str, manifest_sha256: str | None,
    result: dict | None = None, error_type: str | None = None,
) -> str:
    """Create one durable, bounded inference report beside SHA snapshot.

    Never record a prompt, raw error message, secret, checkpoint path, or
    parameter payload. An existing result file always fails closed.
    """
    import json
    import os
    from pathlib import Path

    if status not in {"completed", "failed"}:
        raise ValueError("invalid experimental result status")
    target = Path(str(manifest_path) + ".inference.json")
    if (not target.is_absolute() or target.exists() or target.is_symlink()
            or not target.parent.is_dir()):
        raise ValueError("experiment result output must be a new absolute path")
    data = {
        "status": status,
        "phase": "pretrained_source_exploratory",
        "checkpoint_integrity": "unreviewed_stream_hashes",
        "independent_source_provenance_verified": False,
        "manifest_sha256": manifest_sha256,
        "error_type": error_type if status == "failed" else None,
    }
    if status == "completed":
        if not isinstance(result, dict):
            raise ValueError("successful experiment must supply inference output")
        data.update({
            "full_model": result.get("full_model"),
            "physical_layers": result.get("physical_layers"),
            "all_layers_executed": result.get("all_layers_executed"),
            "input_tokens": result.get("input_tokens"),
            "generated_ids": result.get("generated_ids"),
            "completion": result.get("completion"),
            "finish_reason": result.get("finish_reason"),
            "heldout_quality_verified": False,
        })
    payload = json.dumps(data, sort_keys=True, indent=2).encode() + b"\n"
    with target.open("xb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    return str(target)
