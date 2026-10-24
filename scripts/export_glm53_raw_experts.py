#!/usr/bin/env python3
"""Export exact official GLM-5.3 BF16 selected MoE weights without attention.

Default: small smoke export (first routed stage: router, shared, ONE expert).
Use --full explicitly for all 23 sparse stages x 64/128/256 selected experts.
The output is deliberately NOT a runnable IQ checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from iq_transfer.glm53 import GLM53Inspector
from iq_transfer.glm53_raw_expert_export import (
    RawExpertExportError,
    export_raw_selected_experts,
    validate_export_plan,
)
from iq_transfer.http_range_source import HubSafetensorsSource
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True,
                   help="verified pinned GLM selective shard plan JSON")
    p.add_argument("--index", type=Path, required=True,
                   help="official model.safetensors.index.json (282 shards)")
    p.add_argument("--config", type=Path, required=True,
                   help="official GLM-5.3-BF16 config.json")
    p.add_argument("--output", type=Path, required=True,
                   help="new empty directory for the donor-only bundle")
    p.add_argument("--full", action="store_true",
                   help="export ALL selected expert weights (large / long-running)")
    args = p.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        index_bytes = args.index.read_bytes()
        index = json.loads(index_bytes)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(plan, dict) or not isinstance(weight_map, dict):
            raise RawExpertExportError("invalid plan or official safetensors index")
        if not isinstance(config, dict):
            raise RawExpertExportError("invalid official donor config")
        inspector = GLM53Inspector.from_config_mapping(config)
        layout = inspector.layout
        if (
            config.get("model_type") != "glm_moe_dsa"
            or inspector.config.num_hidden_layers != 78
            or layout.n_routed_experts != 256
            or layout.first_k_dense_replace != 3
            or config.get("quantization_config") is not None
            or len(set(weight_map.values())) != 282
        ):
            raise RawExpertExportError("not the official GLM-5.3 282-shard BF16 index/config")
        if str(inspector.config.dtype).lower() not in (
            "bfloat16", "bf16", "torch.bfloat16"
        ):
            raise RawExpertExportError("official donor must be BF16")
        tasks = validate_export_plan(
            plan, weight_map,
            first_dense=layout.first_k_dense_replace,
            source_expert_count=layout.n_routed_experts,
        )
        # Use local *metadata* only. Place a copy of the official index in a
        # temporary source directory so the range reader never sees any
        # unrelated/nonofficial local shard files.
        with tempfile.TemporaryDirectory(prefix="iq-glm-expert-index-") as tmp:
            metadata = Path(tmp)
            (metadata / "model.safetensors.index.json").write_bytes(index_bytes)
            source = HubSafetensorsSource(
                metadata,
                repo_id=REPO,
                revision=REVISION,
            )
            try:
                report = export_raw_selected_experts(
                    source=source,
                    plan=plan,
                    tasks=tasks,
                    root=args.output,
                    hidden=inspector.config.hidden_size,
                    intermediate=layout.moe_intermediate_size,
                    index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                    max_stages=None if args.full else 1,
                    max_experts=None if args.full else 1,
                )
            finally:
                source.close()
        print(json.dumps({
            "status": report["status"],
            "tensor_count": report["tensor_count"],
            "stages_exported": len(report["layers"]),
            "selected_experts_exported": sum(
                layer["exported_count"] for layer in report["layers"]
            ),
            "output_dir": str(args.output.resolve()),
            "not_an_iq_checkpoint": True,
            "attention_tensors_fetched": 0,
        }, indent=2))
    except (OSError, ValueError, KeyError, RawExpertExportError) as exc:
        p.exit(2, f"GLM raw export failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
