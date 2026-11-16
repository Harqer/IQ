#!/usr/bin/env python3
"""CPU-only structural audit: mixer-only vs full Mamba-3 MIMO block.

This is an AST inspection, not a claim that weight tensors or forward
numerics match the upstream donor. No GPU, weights or dependencies needed.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def audit_hybrid_mamba(source: str) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"status": "blocked", "findings": [f"Invalid source: {exc}"]}

    cls = next((node for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "Mamba3ResidualLayer"), None)
    if cls is None:
        return {"status": "blocked",
                "findings": ["Mamba3ResidualLayer class missing"]}

    def self_attributes(node: ast.AST) -> set[str]:
        attrs: set[str] = set()
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"):
                attrs.add(sub.attr)
        return attrs

    methods = {node.name: node for node in cls.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    init = methods.get("__init__")
    forward = methods.get("forward")
    if init is None or forward is None:
        return {"status": "blocked", "findings": ["Mamba block init/forward missing"]}

    declared = self_attributes(init)
    used = self_attributes(forward)
    missing = []
    for attr in ("mamba", "mlp", "norm", "norm2"):
        if attr not in declared:
            missing.append(f"block_component:{attr}")
    for attr in ("mamba", "mlp"):
        if attr not in used:
            missing.append(f"forward_component:{attr}")

    return {
        "status": "blocked" if missing else "structurally_complete_unverified",
        "class": "Mamba3ResidualLayer",
        "declared": sorted(declared),
        "forward_uses": sorted(used),
        "findings": missing,
        "note": ("An upstream Mamba-3 donor block includes recurrent mixer + gated MLP "
                 "with separate normalization. Structural completeness does NOT "
                 "prove MIMO rank/state/kernel, donor tensor layout or numerical parity."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("iq_model/hybrid.py"))
    args = parser.parse_args()
    try:
        report = audit_hybrid_mamba(args.source.read_text(encoding="utf-8"))
    except OSError as exc:
        report = {"status": "blocked", "findings": [f"Cannot read source: {exc}"]}
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["status"] == "blocked")


if __name__ == "__main__":
    raise SystemExit(main())
