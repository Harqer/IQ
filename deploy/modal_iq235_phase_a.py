"""One-shot Modal GPU run: REAL IQ 235-layer forward, GLM tokenizer.

Before use:
    python3 -m scripts.iq235_preflight
    modal setup
    # Set Modal workspace spend limit and budget in Usage & Billing first.
    modal run deploy/modal_iq235_phase_a.py

No permanent deployment, no web endpoint, no --detach. This run does not
load the existing checkpoint weights; Phase A initializes the full topology.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
STAGED = REPO / "deploy" / "_iq235_baseten_packages"

if modal.is_local() and (
    not STAGED.is_dir() or not (STAGED / "stage_manifest.json").is_file()
):
    raise RuntimeError(
        "Missing verified IQ source stage; run "
        "python3 -m scripts.iq235_preflight from the repository first"
    )

# Copy only the 44 verified source/config files staged by IQ, not HF weights,
# secrets, local virtualenvs or the entire repository.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"
    )
    .apt_install("git")
    .pip_install(
        "transformers==5.17.0",
        "huggingface_hub>=0.35",
        "safetensors>=0.4.5",
        "numpy>=1.26",
        "einops",
        "ninja",
        "packaging",
        "tilelang==0.1.8",
        "apache-tvm-ffi==0.1.9",
        "quack-kernels",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
    )
    .add_local_dir(STAGED, remote_path="/root/IQ", copy=True)
    .run_commands(
        "git clone https://github.com/state-spaces/mamba.git /tmp/iq-mamba3-src",
        "git -C /tmp/iq-mamba3-src checkout e9594ce1c732d97440f0332fdc43170a2294dbfa",
        "cd /root/IQ && python -m scripts.patch_iq_native_mimo_tail --patch-checkout /tmp/iq-mamba3-src",
        "TORCH_CUDA_ARCH_LIST=9.0 MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps /tmp/iq-mamba3-src",
    )
)

app = modal.App("iq235-glm53-architecture-one-shot")

# Hard per-call timeout. Modal bills GPU time during startup/initialization as
# well, so set a Workspace spend limit of $0 beyond credits AND usage budget.
# No persistent service, no concurrent replicas and fast scale-to-zero.
@app.function(
    image=image,
    gpu="H200:8",
    cpu=16,
    memory=131072,
    timeout=480,
    startup_timeout=180,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    single_use_containers=True,
)
def full_iq_phase_a_once() -> dict:
    import sys
    import time
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    sys.path.insert(0, "/root/IQ")
    os.environ["IQ_PHASE"] = "architecture_only"
    from iq_cloud.full_iq235 import load_phase_a, generate_probe

    provider = TracerProvider(resource=Resource.create({
        "service.name": "iq235-modal",
        "iq.phase": "architecture_only",
        "iq.full_model": "true",
    }))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("iq235.modal.full")

    started = time.monotonic()
    try:
        with tracer.start_as_current_span("iq235.modal.load_complete_architecture"):
            state = load_phase_a()
        with tracer.start_as_current_span("iq235.modal.generate_one_token"):
            report = generate_probe(
                state,
                "The next integer after 1 2 3 is",
                max_new_tokens=1,
                tracer=tracer,
            )
        report["backend"] = "modal"
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report
    finally:
        provider.force_flush(timeout_millis=5000)
        provider.shutdown()


@app.local_entrypoint()
def main():
    result = full_iq_phase_a_once.remote()
    if not (result.get("full_model") is True
            and result.get("physical_layers") == 235
            and result.get("all_layers_executed") is True
            and result.get("glm_tokenizer_vocab") == 154880
            and result.get("checkpoint_weights_injected") is False
            and result.get("generated_ids")):
        raise RuntimeError("Modal result did not prove the complete IQ Phase-A forward")

    location = REPO / "artifacts" / "iq235-modal"
    location.mkdir(parents=True, exist_ok=True)
    path = location / "architecture_only.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite old evidence: {path}")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"FULL_IQ235_MODAL_FORWARD_PASS: {path}", flush=True)
