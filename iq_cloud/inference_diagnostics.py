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
