#!/usr/bin/env python3
"""Validate the full active GLM-5.3 transfer toolchain before downloading weights.

Checks installed packages, symbol exports, and all operational entrypoints.
Does not call their main() functions, download files, start GPU, or modify data.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REQUIRED = {
    "numpy": "numpy>=1.26",
    "torch": "CPU PyTorch already in .venv-iq-client; do not reinstall CUDA wheels",
    "safetensors": "safetensors>=0.4.5",
    "huggingface_hub": "huggingface_hub>=0.35",
    "requests": "requests>=2.32,<3",
}
ENTRYPOINTS = (
    "iq_model",
    "iq_transfer",
    "iq_transfer.glm53",
    "iq_transfer.glm53_job",
    "iq_transfer.glm53_calibration",
    "iq_transfer.glm53_compile",
    "iq_transfer.glm53_shards",
    "iq_transfer.glm53_stream_capture",
    "iq_transfer.foundation_reuse",
    "scripts.plan_glm53_weight_extension",
    "scripts.plan_glm53_selective_shards",
    "scripts.estimate_glm53_shard_bytes",
    "scripts.probe_glm53_remote_ranges",
    "scripts.reuse_glm53_calibration",
    "scripts.verify_mamba_frozen",
    "scripts.check_glm53_source_inventory",
)

def check_environment() -> dict:
    root = Path(__file__).resolve().parents[1]
    problems = []
    if sys.prefix == sys.base_prefix:
        problems.append("No active virtualenv: source ~/IQ-integrate/.venv-iq-client/bin/activate")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    installed = {}
    for module, requirement in REQUIRED.items():
        try:
            mod = importlib.import_module(module)
            installed[module] = getattr(mod, "__version__", "present")
        except Exception as exc:
            problems.append(f"Package {module} missing/broken ({requirement}): {type(exc).__name__}: {exc}")
    loaded = []
    for name in ENTRYPOINTS:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as exc:
            problems.append(f"Module {name} import failed: {type(exc).__name__}: {exc}")
    if "huggingface_hub" in installed:
        try:
            hub = importlib.import_module("huggingface_hub")
            for symbol in ("HfFileSystem", "HfApi", "get_token", "hf_hub_url", "hf_hub_download"):
                if not hasattr(hub, symbol):
                    problems.append(f"huggingface_hub missing required symbol: {symbol}")
        except Exception as exc:
            problems.append(f"Hugging Face API mismatch: {type(exc).__name__}: {exc}")
    return {
        "interpreter": sys.executable,
        "virtualenv_active": sys.prefix != sys.base_prefix,
        "packages": installed,
        "entrypoints_imported": len(loaded),
        "entrypoints_total": len(ENTRYPOINTS),
        "problems": problems,
        "status": "ready_for_glm53_metadata_workflow" if not problems else "missing_or_broken_imports",
        "note": (
            "Does not validate actual donor weights or optional transformers runtime "
            "for model activation capture. No GPU workload launched."
        ),
    }

def main() -> int:
    report = check_environment()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready_for_glm53_metadata_workflow" else 1

if __name__ == "__main__":
    raise SystemExit(main())
