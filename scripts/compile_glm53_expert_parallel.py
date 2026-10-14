#!/usr/bin/env python3
"""Compile real GLM-5.3 BF16 expert tensors into IQ expert-parallel shards.

This requires pinned, fully staged GLM safetensors and actual activation-
refined coordinate calibration. No weights are fabricated or downloaded.
"""
from __future__ import annotations

import argparse
from iq_transfer.glm53_expert_shards import compile_glm53_expert_shard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-checkpoint", required=True,
                        help="mounted local GLM HF-format BF16 checkpoint directory")
    parser.add_argument("--calibration", required=True,
                        help="actual 78-stage IQ activation-refined calibration directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--donor-revision", required=True,
                        help="verified full Git commit SHA for staged GLM donor")
    parser.add_argument("--ep-rank", required=True, type=int)
    parser.add_argument("--ep-size", required=True, type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report = compile_glm53_expert_shard(
        donor_dir=args.glm_checkpoint, calibration_dir=args.calibration,
        output_dir=args.output, ep_rank=args.ep_rank, ep_size=args.ep_size,
        donor_revision=args.donor_revision, device=args.device,
    )
    print(
        f"EP rank {report['ep_rank']}/{report['ep_size']}: "
        f"{report['local_expert_shards']} complete expert shards, "
        f"{len(report['files'])} total checksummed files"
    )
    print("Attention, native Mamba and final model parity remain unconverted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
