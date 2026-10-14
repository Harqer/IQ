#!/usr/bin/env python3
"""Audit real GLM-5.3 donor tensor indices/headers before any conversion.

Read-only to donor weights; writes a new local JSON report only. For native IQ
expert transport use the actual zai-org/GLM-5.3-BF16 publisher artifact,
not the default quantized E4M3 FP8 checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iq_transfer.glm53_donor_preflight import audit_glm53_donor
from iq_transfer.glm53_expert_mapping import glm53_expert_transfer_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--identity", required=True,
        choices=("zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16")
    )
    parser.add_argument("--ep-size", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-bf16", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.exists():
        parser.error("refuse to overwrite a prior donor audit report")
    result = audit_glm53_donor(
        args.checkpoint,
        checkpoint_identity=args.identity,
        require_bf16=args.require_bf16,
        verify_headers=True,
    )
    plan = glm53_expert_transfer_manifest(args.ep_size)
    audit = {
        "donor": result.to_dict(),
        "iq_expert_mapping_sha256": plan["sha256"],
        "iq_expert_mapping_tasks": len(plan["tasks"]),
        "conversion_executed": False,
        "complete_glm5bridge_model_conversion": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"GLM donor audit written: {destination}")
    print(f"Required source keys: {result.required_keys:,}")
    print(f"Rank-aware IQ expert mappings: {len(plan['tasks']):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
