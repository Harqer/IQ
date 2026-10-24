#!/usr/bin/env python3
"""Build one IQ Stage-2 training example from REAL GLM activations + byte spans.

Input GLM safetensors must contain exactly glm_hidden [GLM-token, GLM-hidden].
Spans JSON: [[byte_start, byte_end], ...]. IQ IDs JSON: [int, ...].
Uses a previously fitted, held-out validated residual_out CoordinateMap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file

from iq_transfer.stage2_artifacts import write_stage2_example
from iq_transfer.transport import load_coordinate_map


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--document", type=Path, required=True, help="identical UTF-8 bytes used for both tokenizations")
    p.add_argument("--glm-hidden", type=Path, required=True, help="real teacher capture; safetensors key glm_hidden")
    p.add_argument("--glm-spans", type=Path, required=True)
    p.add_argument("--iq-spans", type=Path, required=True)
    p.add_argument("--iq-ids", type=Path, required=True)
    p.add_argument("--map-prefix", type=Path, required=True)
    p.add_argument("--source-layer", type=int, required=True)
    p.add_argument("--target-layer", type=int, required=True)
    p.add_argument("--teacher-revision", required=True)
    p.add_argument("--split", choices=("train", "validation"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        captured = load_file(str(a.glm_hidden), device="cpu")
        if set(captured) != {"glm_hidden"}:
            raise ValueError("teacher safetensors must contain only glm_hidden")
        hidden = captured["glm_hidden"].float().numpy()
        source_spans = json.loads(a.glm_spans.read_text(encoding="utf-8"))
        target_spans = json.loads(a.iq_spans.read_text(encoding="utf-8"))
        target_ids = json.loads(a.iq_ids.read_text(encoding="utf-8"))
        if not all(
            isinstance(row, list) and len(row) == 2
            and all(type(value) is int for value in row)
            for rows in (source_spans, target_spans) for row in rows
        ):
            raise ValueError("spans must be integer [start,end] JSON arrays")
        if not isinstance(target_ids, list) or not all(
            type(x) is int for x in target_ids
        ):
            raise ValueError("IQ IDs must be an integer JSON array")
        info = write_stage2_example(
            a.out_dir,
            document_bytes=a.document.read_bytes(),
            glm_states=hidden,
            glm_token_spans=[tuple(row) for row in source_spans],
            iq_token_spans=[tuple(row) for row in target_spans],
            iq_token_ids=target_ids,
            residual_out_map=load_coordinate_map(a.map_prefix),
            source_glm_layer=a.source_layer,
            target_iq_layer=a.target_layer,
            teacher_revision=a.teacher_revision,
            split=a.split,
        )
        print(
            f"Saved {info['causally_aligned_tokens']} causally aligned IQ tokens "
            f"({a.split}) to {a.out_dir}"
        )
        print("Teacher capture authenticity and language quality NOT yet proven.")
        return 0
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Alignment build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
