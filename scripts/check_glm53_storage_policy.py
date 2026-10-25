#!/usr/bin/env python3
"""Fail closed on incorrect IQ full78 output bucket visibility/capacity.

Private buckets require provider-verified space. Public buckets are accepted
ONLY as explicit best-effort destinations; this does not imply 600+ GB of
free storage is guaranteed or authorize billable GPU compute.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import bucket_info

from scripts.full78_storage_preflight import required_private_bytes
from scripts.glm53_full78_gate import check_capacity


class StoragePolicyError(RuntimeError):
    pass


DEFAULT_BUCKET = "kofecheckmate/iq-glm53-all256"


def verify_bucket_policy(
    bucket: str,
    *,
    allow_public_best_effort: bool = False,
    private_attestation: str | None = None,
) -> dict[str, object]:
    # Always inspect the ACTUAL provider state; never trust a caller-supplied
    # claim that its target bucket is public or adequately provisioned.
    if bucket != DEFAULT_BUCKET:
        raise StoragePolicyError("unexpected output bucket; explicit review required")
    try:
        info = bucket_info(bucket)
    except Exception as exc:
        raise StoragePolicyError("cannot verify target bucket visibility") from exc
    if info.private:
        if not private_attestation:
            raise StoragePolicyError(
                "private full78 output requires provider-verified capacity"
            )
        checked = check_capacity(
            Path(private_attestation), required_private_bytes(), bucket
        )
        return {
            "bucket": bucket, "visibility": "private",
            "policy": "verified_private_capacity",
            "verified_free_bytes": checked["free_bytes"],
        }
    if not allow_public_best_effort:
        raise StoragePolicyError(
            "public output is best-effort and requires "
            "IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT=1"
        )
    return {
        "bucket": bucket,
        "visibility": "public",
        "policy": "best_effort_not_guaranteed",
        "required_expected_full78_bytes": required_private_bytes(),
        "warnings": [
            "Hugging Face may reject large public uploads under fair-use policies",
            "Existing partial artifacts remain unchanged until an explicit writer starts",
            "This only checks STORAGE; HF GPU Jobs remain BILLABLE",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    a = p.parse_args()
    try:
        report = verify_bucket_policy(
            a.bucket,
            allow_public_best_effort=os.environ.get(
                "IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT"
            ) == "1",
            private_attestation=os.environ.get("IQ_STORAGE_ATTESTATION"),
        )
    except (StoragePolicyError, OSError, ValueError) as exc:
        p.exit(3, f"FULL78_BUCKET_POLICY_BLOCKED: {exc}\n")
    print("FULL78_OUTPUT_BUCKET_POLICY=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
