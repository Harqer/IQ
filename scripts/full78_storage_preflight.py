"""Fail closed on insufficient private checkpoint capacity before paid GPU work."""
from __future__ import annotations

import argparse
import json
import os

# Observed actual size of one BF16 expert safetensors from the official
# 78-layer job. This is a checkpoint-space lower-bound, not an allocation quote.
EXPERT_FILE_BYTES = 25_166_192
GLM_LAYER_COUNT = 78
ROUTED_PER_LAYER = 256
DENSE_GLMLAYERS = 3
ROUTED_GLMLAYERS = GLM_LAYER_COUNT - DENSE_GLMLAYERS
EXPERT_SLOT_COUNT = GLM_LAYER_COUNT * ROUTED_PER_LAYER
REAL_ROUTED_SLOT_COUNT = ROUTED_GLMLAYERS * ROUTED_PER_LAYER
# Calibration + context + Mamba/native + upload overhead; verify exact quota separately.
NON_EXPERT_RESERVE_BYTES = 45_000_000_000
SAFETY_MARGIN = 1.15


def required_private_bytes() -> int:
    return int((EXPERT_SLOT_COUNT * EXPERT_FILE_BYTES +
                NON_EXPERT_RESERVE_BYTES) * SAFETY_MARGIN)


def verify_capacity(confirmed_free_bytes: int) -> None:
    if confirmed_free_bytes < required_private_bytes():
        raise RuntimeError(
            f"FULL78_STORAGE_BLOCKED: {confirmed_free_bytes/1e9:.1f} GB "
            f"available, minimum {required_private_bytes()/1e9:.1f} GB. "
            "Do not start a paid job until the destination has enough private storage."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmed-free-gb",type=float,
                   default=float(os.environ.get("IQ_CONFIRMED_FREE_PRIVATE_GB", "0")))
    a=p.parse_args()
    print(json.dumps({
        "real_glm_routed_slots": REAL_ROUTED_SLOT_COUNT,
        "all_iq_routed_slots": EXPERT_SLOT_COUNT,
        "bf16_expert_files_gb": round(EXPERT_SLOT_COUNT * EXPERT_FILE_BYTES/1e9,2),
        "required_free_private_gb": round(required_private_bytes()/1e9,1),
        "verified_available_gb":a.confirmed_free_gb,
    },sort_keys=True),flush=True)
    verify_capacity(int(a.confirmed_free_gb*1e9))
    print("FULL78_PRIVATE_STORAGE_PREFLIGHT_PASSED",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
