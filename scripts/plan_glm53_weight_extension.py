#!/usr/bin/env python3
"""CPU-only GLM-5.3 donor-coverage inventory and immutable expansion plan.

Grounded in a real GLM config, IQ checkpoint manifest, and IQ config.
No donor weights, CUDA, paid endpoints or synthetic expert identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


class CoverageError(ValueError):
    pass


def plan_coverage(
    donor: dict, transplant: dict, recipient: dict, *, target_experts: int = 64
) -> dict:
    if type(target_experts) is not int or target_experts not in (32, 64, 128, 256):
        raise CoverageError("target routed experts must be one of 32, 64, 128, 256")
    if donor.get("model_type") != "glm_moe_dsa":
        raise CoverageError("expected original GLM-5.3 DSA/MoE donor")
    if transplant.get("capability_source", {}).get("repo") != "zai-org/GLM-5.3-BF16":
        raise CoverageError("GLM source provenance mismatch")
    if transplant.get("foundation", {}).get("repo") != "state-spaces/mamba3-mimo-1.5b":
        raise CoverageError("Mamba-3 foundation provenance mismatch")
    if transplant.get("recipient_config_fingerprint") is None:
        raise CoverageError("recipient checkpoint lacks config fingerprint")
    n = donor.get("num_hidden_layers")
    first_dense = donor.get("first_k_dense_replace")
    source_experts = donor.get("n_routed_experts")
    if not all(type(v) is int for v in (n, first_dense, source_experts)):
        raise CoverageError("source model architecture is incomplete")
    if not 0 <= first_dense < n or not 0 < source_experts:
        raise CoverageError("invalid source model configuration")
    if source_experts != 256 or n != 78:
        raise CoverageError("expected pinned GLM-5.3 78-layer/256-expert layout")
    old = recipient.get("stable_moe", {}).get("num_experts")
    if old != 32:
        raise CoverageError("source checkpoint is not the known 32-expert starting point")
    if recipient.get("model", {}).get("vocab_size") != 128256:
        raise CoverageError("IQ must keep Mamba lexical vocabulary")
    schedule = recipient.get("schedule", {}).get("layers")
    if not isinstance(schedule, list):
        raise CoverageError("IQ schedule missing")
    layers = Counter(schedule)
    if len(schedule) != 80 or layers != {"mamba3": 32, "csa": 12, "hca": 12, "moe": 24}:
        raise CoverageError("IQ's physical Mamba/context/MoE schedule differs from baseline")
    raw = transplant.get("source_layer_map")
    if not isinstance(raw, dict) or len(raw) != layers["moe"]:
        raise CoverageError("source-layer map missing 24 GLM stage records")
    try:
        keys = sorted((int(key), value) for key, value in raw.items())
    except (TypeError, ValueError) as e:
        raise CoverageError("invalid source layer-map stage indices") from e
    if [key for key, _ in keys] != list(range(24)):
        raise CoverageError("source stage ordinals not contiguous")
    selected = [value for _, value in keys]
    if any(type(v) is not int or v < 0 or v >= n for v in selected):
        raise CoverageError("source-layer map contains invalid GLM layer indices")
    if selected != sorted(set(selected)):
        raise CoverageError("source-layer map must be monotonic and unique")
    chosen = set(selected)
    dense = [i for i in range(first_dense) if i in chosen]
    sparse = [i for i in range(first_dense, n) if i in chosen]
    omitted_layers = [i for i in range(n) if i not in chosen]
    source_ids = transplant.get("selected_source_expert_ids")
    exact = isinstance(source_ids, dict) and len(source_ids) == 24
    if exact:
        for ordinal, source_layer in enumerate(selected):
            ids = source_ids.get(str(ordinal))
            if source_layer < first_dense:
                if ids is not None:
                    raise CoverageError("dense stage cannot have routed expert IDs")
            elif (
                not isinstance(ids, list)
                or len(ids) != old
                or any(type(i) is not int or i < 0 or i >= source_experts for i in ids)
                or len(set(ids)) != old
            ):
                raise CoverageError("expert identity provenance malformed")
    remaining_sufficient = source_experts >= target_experts
    if not remaining_sufficient:
        raise CoverageError("target requests more experts than GLM donor has")
    initial_slots = len(sparse) * old
    planned_slots = len(sparse) * target_experts
    source_slots = (n - first_dense) * source_experts
    return {
        "schema_version": 1,
        "status": "plan_only_no_weights_transferred",
        "donor": {"repo": transplant["capability_source"]["repo"],
                  "revision": transplant["capability_source"].get("revision"),
                  "layers": n, "first_dense_layers": first_dense,
                  "routed_experts_per_sparse_layer": source_experts},
        "recipient": {"original_experts_per_sparse_stage": old,
                      "planned_experts_per_sparse_stage": target_experts,
                      "mamba_layers": layers["mamba3"],
                      "context_layers": layers["csa"] + layers["hca"],
                      "moe_layers": layers["moe"], "unchanged_lexical_vocab": 128256,
                      "mamba_weights_must_remain_identical": True},
        "selected_source_layers": selected,
        "not_yet_mapped_source_layers": omitted_layers,
        "covered_dense_source_layers": dense,
        "unmapped_dense_source_layers": [
            i for i in range(first_dense) if i not in chosen
        ],
        "covered_sparse_source_layers": sparse,
        "unmapped_sparse_source_layers": [
            i for i in range(first_dense, n) if i not in chosen
        ],
        "routed_expert_slots": {
            "source_total": source_slots,
            "original_mapped_max": initial_slots,
            "planned_mapped_max": planned_slots,
            "new_slots": planned_slots - initial_slots,
            "original_fraction": initial_slots / source_slots,
            "planned_fraction": planned_slots / source_slots,
        },
        "expert_ids_in_existing_checkpoint_verified": exact,
        "expert_identity_note": (
            "Exact selected donor IDs recorded and schema-checked"
            if exact else
            "The old checkpoint has no expert IDs: exact identities and "
            "preservation of prior expert weight tensors cannot be certified "
            "until selection provenance and bytewise hashes are compared."
        ),
        "next_steps": [
            "Rebootstrap GLM calibration with matching routed-expert capacity",
            "Compile into a NEW output directory; never overwrite original mocha bucket",
            "Verify exact original Mamba tensor bytes and model-global lexical tensors",
            "Audit selected expert identities and per-stage residual scales",
            "Evaluate coherence; passing load/shape coverage alone is not enough",
        ],
    }


def read_source(ref: str) -> dict:
    if ref.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        with fs.open(ref, "r") as stream:
            return json.load(stream)
    return json.loads(Path(ref).read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor-config", default="hf://models/zai-org/GLM-5.3-BF16/config.json")
    p.add_argument("--transplant-manifest", default="hf://buckets/kofecheckmate/mocha/transplant_manifest.json")
    p.add_argument("--iq-config", default="hf://buckets/kofecheckmate/mocha/iq_config.json")
    p.add_argument("--routed-experts", type=int, choices=(32, 64, 128, 256), default=64)
    p.add_argument("--output", type=Path, help="optional NEW JSON output path")
    args = p.parse_args()
    try:
        result = plan_coverage(
            read_source(args.donor_config),
            read_source(args.transplant_manifest),
            read_source(args.iq_config),
            target_experts=args.routed_experts,
        )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(payload)
        if args.output:
            if args.output.exists():
                raise CoverageError("output already exists; no overwrite permitted")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"GLM expansion planning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
