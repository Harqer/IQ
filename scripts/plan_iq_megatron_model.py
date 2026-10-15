#!/usr/bin/env python3
"""Write source-verified IQ HF config and an MCore pipeline ownership audit.

This is a CPU-only architecture compatibility gate. No model weights, custom
MCore ModelBuilder, or Megatron conversion are silently synthesized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iq_training.hf_model_config import IQHFCausalConfig
from iq_training.pipeline_contract import partition_iq_78
from iq_transfer.complete_transplant import nvidia_parallel_78_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument(
        "--pp-cuts", default="0,78",
        help="comma-separated global layer boundaries, e.g. 0,6,10,78"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        cuts = tuple(int(s.strip()) for s in args.pp_cuts.split(","))
    except ValueError:
        parser.error("--pp-cuts must contain comma-separated integer layer boundaries")
    config = nvidia_parallel_78_config(
        expert_parallel_size=args.ep_size
    )
    plan = partition_iq_78(config, boundaries=cuts)
    output = args.output.resolve()
    if output.exists():
        parser.error("refusing to overwrite existing model contract output")
    output.mkdir(parents=True, exist_ok=False)
    (output / "hf").mkdir(exist_ok=False)
    hf = IQHFCausalConfig(iq_hybrid=config.to_dict())
    hf.save_pretrained(output / "hf")
    manifest = plan.to_dict()
    manifest["hf_model_type"] = hf.model_type
    manifest["hf_config_path"] = "hf/config.json"
    manifest["source_of_truth"] = [
        "https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html",
        "https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/pipeline_parallel_layout.html",
    ]
    path = output / "pipeline-contract.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved IQ architecture and pipeline contract to {output}")
    print(f"Blocks: 78; expert parallel: {args.ep_size}; pipeline ranks: {plan.pp_size}")
    print(f"Executable as native Megatron Bridge model: no")
    for blocker in plan.blocked_capabilities:
        print(f"Blocked: {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
