#!/usr/bin/env python3
"""Verify the real GLM-5.3 donor topology against the IQ 78-fused-block target.

Reads the mounted donor's actual config.json (no model tensors). A successful
preflight is *not* a weight transfer or numerical-equivalence validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iq_transfer.glm53_parallel_contract import GLM53Topology
from iq_transfer.complete_transplant import nvidia_parallel_78_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite existing GLM transfer manifest")
    donor = json.loads(args.donor_config.read_text(encoding="utf-8"))
    topology = GLM53Topology.from_hf_config(donor)
    iq = nvidia_parallel_78_config()
    if iq.parallel_dense_layers != topology.dense_blocks:
        raise RuntimeError("IQ dense stages do not match pinned GLM checkpoint")
    if iq.parallel_index_sources != topology.index_source_blocks:
        raise RuntimeError("IQ IndexShare routing does not match pinned GLM checkpoint")
    if iq.stable_moe is None or (
        iq.stable_moe.num_experts != topology.num_routed_experts
        or iq.stable_moe.top_k != topology.top_k
    ):
        raise RuntimeError("IQ expert geometry does not match GLM checkpoint")
    payload = topology.export_manifest()
    payload["iq_config_fingerprint"] = iq.fingerprint
    payload["validation"] = "geometry_only_no_tensors_loaded"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(payload['stages'])} GLM blocks; IQ output {args.output}")
    print("No pretrained weights were converted or uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
