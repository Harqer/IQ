#!/usr/bin/env python3
"""IQ's official Megatron Bridge donor converter and MSC multi-bucket entrypoint.

No model or bucket is modified by default. Use 'import --execute' only on a
worker with the real Megatron Bridge installation and sufficient resources.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from iq_transfer.megatron_bridge import (
    DONOR_PINS, execute_import, make_import_plan,
    probe_provider, render_plan,
)
from iq_transfer.megatron_storage import (
    load_profile_spec, mocha_profile, render_msc_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    storage = sub.add_parser("storage-config", help="generate MSC YAML for Mocha plus arbitrary cloud buckets")
    storage.add_argument("--output", required=True)
    storage.add_argument(
        "--additional-profiles-json",
        help="optional JSON spec with an array of MSC profiles (s3/gcs/azure/oci/ais/s8k)",
    )
    storage.add_argument("--cache-dir", required=True)
    storage.add_argument("--cache-size", default="500G")

    imp = sub.add_parser("import", help="NVIDIA Nemotron/GLM donor HF-to-Megatron import")
    imp.add_argument("--donor", choices=tuple(DONOR_PINS), required=True)
    imp.add_argument(
        "--source", required=True,
        help="pinned HF repo ID or absolute path to a mounted Hub-format checkpoint",
    )
    imp.add_argument("--output", required=True, help="new local Megatron checkpoint directory")
    imp.add_argument("--bridge-root", required=True, help="absolute Megatron-Bridge checkout")
    imp.add_argument("--device", choices=("gpu", "cpu"), required=True)
    imp.add_argument("--executor", choices=("local", "slurm"), default="local")
    imp.add_argument("--gpus-per-node", type=int, default=0)
    imp.add_argument("--nodes", type=int, default=1)
    imp.add_argument("--cpu-processes-per-node", type=int, default=1)
    imp.add_argument("--tp", type=int, default=1)
    imp.add_argument("--pp", type=int, default=1)
    imp.add_argument("--ep", type=int, default=1)
    imp.add_argument("--etp", type=int, default=1)
    imp.add_argument("--timeout-minutes", type=int, default=60)
    imp.add_argument("--manifest", help="optional local output for immutable donor import plan")
    imp.add_argument("--probe-provider", action="store_true",
                     help="run real AutoBridge metadata/provider inspection, without weights")
    imp.add_argument("--execute", action="store_true",
                     help="execute official convert.sh import rather than only generating a plan")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "storage-config":
        profiles = [mocha_profile()]
        if args.additional_profiles_json:
            profiles.extend(load_profile_spec(args.additional_profiles_json))
        text = render_msc_config(
            profiles, cache_dir=args.cache_dir, cache_size=args.cache_size
        )
        output = Path(args.output)
        if output.exists():
            raise SystemExit("refusing to overwrite MSC configuration")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"MSC_CONFIG={output}")
        print("Contains environment VARIABLE NAMES only; export credentials on your worker.")
        return 0

    plan = make_import_plan(
        donor=args.donor, source=args.source, output_dir=args.output,
        bridge_root=args.bridge_root, device=args.device, executor=args.executor,
        gpus_per_node=args.gpus_per_node, tensor_parallel=args.tp,
        pipeline_parallel=args.pp, expert_parallel=args.ep,
        expert_tensor_parallel=args.etp, nodes=args.nodes,
        cpu_processes_per_node=args.cpu_processes_per_node,
        distributed_timeout_minutes=args.timeout_minutes,
    )
    if args.manifest:
        destination = Path(args.manifest)
        if destination.exists():
            raise SystemExit("refusing to overwrite Bridge import manifest")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_plan(plan), encoding="utf-8")
    print(render_plan(plan))
    if args.probe_provider:
        print(probe_provider(plan))
    if args.execute:
        execute_import(plan)
        print("Bridge completed a donor conversion; IQ fusion conversion has NOT been executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
