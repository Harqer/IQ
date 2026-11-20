"""One IQ235 readiness decision: no GPUs, no weight payload downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iq_cloud.bulk_readiness import remote_header_audit, static_release_blockers
from iq_cloud.full235_weight_loader import PINNED_SOURCE_SHARD_MANIFEST_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-header-samples", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = static_release_blockers(
        root, approved_digest_pin=PINNED_SOURCE_SHARD_MANIFEST_SHA256,
    )
    report["representative_source_headers"] = (
        remote_header_audit() if args.remote_header_samples else "not_requested"
    )
    report["gpu_started"] = False
    report["weight_payloads_downloaded"] = False
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
