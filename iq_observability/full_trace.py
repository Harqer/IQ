"""Full-weight IQ inference tracing. No model substitutions or simulated execution."""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import torch

EXPECTED = {"mamba3": 32, "csa": 12, "hca": 12, "moe": 24}


def _otel_tracer():
    """OTLP export is opt-in. Local JSONL traces always remain available."""
    if not (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None, None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("OTLP requested but OpenTelemetry exporter is missing") from exc
    provider = TracerProvider(resource=Resource.create({"service.name": "iq-full-weight-inference"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("iq.full_inference"), provider


class IQTrace:
    """One real IQ request: stage timings, CUDA memory, all 80 layers and MoE routes."""

    def __init__(self, directory: str | Path = "/tmp/iq-traces"):
        self.run_id = uuid.uuid4().hex
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.path = target / f"{self.run_id}.jsonl"
        self._out = self.path.open("a", encoding="utf-8", buffering=1)
        self._tracer, self._provider = _otel_tracer()
        self._handles = []
        self._started = {}
        self._contexts = {}
        self._completed = Counter()
        self._schedule = []
        self._sync = os.getenv("IQ_TRACE_SYNC", "1") == "1"

    def event(self, name: str, **attributes):
        record = {"event": name, "run_id": self.run_id, "ts_ns": time.time_ns()}
        record.update(attributes)
        encoded = json.dumps(record, default=str, separators=(",", ":"), sort_keys=True)
        self._out.write(encoded + "\n")
        print("IQ_TRACE " + encoded, file=sys.stderr, flush=True)

    def _memory(self):
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        }

    @contextlib.contextmanager
    def stage(self, name, **attributes):
        self.event(name + ".start", **attributes, **self._memory())
        begin = time.perf_counter_ns()
        span_cm = (
            self._tracer.start_as_current_span(name, attributes={
                "iq.run_id": self.run_id,
                **{f"iq.{k}": v for k, v in attributes.items() if isinstance(v, (int, float, bool, str))},
            })
            if self._tracer else contextlib.nullcontext()
        )
        try:
            with span_cm:
                yield
        except BaseException as exc:
            self.event(name + ".error", exception=type(exc).__name__,
                       message=str(exc)[:300], elapsed_ms=round((time.perf_counter_ns() - begin)/1e6, 3),
                       **self._memory())
            raise
        else:
            self.event(name + ".end", elapsed_ms=round((time.perf_counter_ns()-begin)/1e6, 3),
                       **self._memory())

    def attach(self, model):
        schedule = [kind.value for kind in model.config.schedule.layers]
        if len(schedule) != 80 or dict(Counter(schedule)) != EXPECTED:
            raise RuntimeError(f"Refusing reduced/altered IQ schedule: {Counter(schedule)}")
        if model.config.moe_variant != "stable_latent" or model.config.stable_moe is None:
            raise RuntimeError("Full checkpoint requires Stable LatentMoE")
        if model.config.mamba3.mimo_rank != 4:
            raise RuntimeError("Mamba-3 MIMO rank-4 is mandatory")
        self._schedule = schedule
        self.event("architecture.verified",
                   layers=len(schedule), hidden_size=model.config.model.hidden_size,
                   vocabulary=model.config.model.vocab_size,
                   mamba_rank=model.config.mamba3.mimo_rank,
                   moe_experts=model.config.stable_moe.num_experts,
                   moe_top_k=model.config.stable_moe.top_k)
        named = [(f"layer.{i:02d}.{name}", layer, name)
                 for i, (name, layer) in enumerate(zip(schedule, model.layers, strict=True))]
        for name in ("embed_tokens", "norm", "reasoning", "reasoning_injector", "lm_head"):
            module = getattr(model, name, None)
            if module is not None:
                named.append((name, module, None))
        for idx, (kind, layer) in enumerate(zip(schedule, model.layers, strict=True)):
            for component in ("mamba", "attention", "moe"):
                nested = getattr(layer, component, None)
                if nested is not None:
                    named.append((f"kernel.{idx:02d}.{kind}.{component}", nested, None))
        for label, module, kind in named:
            self._handles.append(module.register_forward_pre_hook(self._before(label)))
            self._handles.append(module.register_forward_hook(
                self._after(label, kind), always_call=True))
        return self

    def round_start(self):
        self._completed.clear()

    def assert_complete(self):
        actual = dict(self._completed)
        if actual != EXPECTED:
            raise RuntimeError(f"Incomplete full-model forward: actual={actual}; expected={EXPECTED}")
        self.event("forward.complete", layers=80, **self._memory())

    def _before(self, label):
        def callback(_module, _inputs):
            if self._sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._started[label] = time.perf_counter_ns()
            cm = self._tracer.start_as_current_span(
                "iq." + label, attributes={"iq.run_id": self.run_id}
            ) if self._tracer else contextlib.nullcontext()
            cm.__enter__()
            self._contexts[label] = cm
        return callback

    def _after(self, label, kind):
        def callback(_module, _inputs, result):
            try:
                if self._sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = round((time.perf_counter_ns() - self._started.get(label, time.perf_counter_ns())) / 1e6, 3)
                if result is None:
                    self.event("module.failed", module=label, elapsed_ms=elapsed, **self._memory())
                    return
                tensor = result if isinstance(result, torch.Tensor) else getattr(result, "hidden_states", None)
                if tensor is None:
                    tensor = getattr(result, "state", None)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"{label}: missing tensor output")
                values = tensor.detach().float()
                finite = bool(torch.isfinite(values).all().item())
                stats = {
                    "module": label, "kind": kind or "component",
                    "elapsed_ms": elapsed, "shape": list(tensor.shape),
                    "finite": finite, "rms": float(values.square().mean().sqrt().item()) if finite else None,
                    "abs_max": float(values.abs().max().item()) if finite else None,
                }
                if kind == "moe":
                    counts = getattr(result, "expert_counts", None)
                    if isinstance(counts, torch.Tensor):
                        stats["expert_assignments"] = int(counts.sum().item())
                        stats["experts_active"] = int((counts > 0).sum().item())
                        stats["expert_count"] = int(counts.numel())
                        if int(counts.numel()) != 32:
                            raise RuntimeError("MoE expert count was not 32")
                self.event("module.complete", **stats, **self._memory())
                if not finite:
                    raise FloatingPointError(f"{label}: NaN/Inf detected")
                if kind:
                    self._completed[kind] += 1
            finally:
                cm = self._contexts.pop(label, None)
                if cm:
                    cm.__exit__(None, None, None)
        return callback

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for cm in list(self._contexts.values()):
            cm.__exit__(None, None, None)
        self._contexts.clear()
        if self._provider:
            self._provider.force_flush(timeout_millis=3000)
            self._provider.shutdown()
        self._out.close()
