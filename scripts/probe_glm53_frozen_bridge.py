#!/usr/bin/env python3
"""Benchmark an exact frozen GLM BF16 expert with trainable low-rank IQ adapters.

Requires pinned, hash-verified stage-1 calibration, original donor weights,
actual teacher pre-MoE states, official router IDs, and the *same* text fixture
that generated the capture. Uses DOCUMENT-disjoint train/validation splits.
No donor, canonical calibration, IQ checkpoint or original report is mutated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import AutoTokenizer
from safetensors.torch import save_file

from iq_transfer.complete_transplant import canonical_glm53_config
from iq_transfer.glm53_basis_functional_compare import (
    _load_baseline_expert,
)
from iq_transfer.glm53_expert_informed_basis import _read_baseline
from iq_transfer.glm53_expert_fidelity import comparison
from iq_transfer.glm53_frozen_bridge import (
    FrozenGLMExpertBridge, train_frozen_bridge,
)
from iq_transfer.glm53_routed_fidelity import read_verified_router_pairs
from iq_transfer.glm53_single_expert_projection import (
    _calibration_map, _json, _read_donor, _sha_file, _sha_tensor,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION
from scripts.project_glm53_single_expert import _restore_stage_from_hf


def _load_document_ids(
    text_file: Path,
    capture_path: Path,
    *,
    max_tokens: int,
    token_count: int,
) -> tuple[np.ndarray, dict[int, int]]:
    lines = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not 1 <= len(lines) <= 8:
        raise ValueError("text fixture must contain 1..8 nonempty documents")
    tokenizer = AutoTokenizer.from_pretrained(
        REPO, revision=REVISION, trust_remote_code=False
    )
    batch = tokenizer(
        lines, return_tensors="pt", truncation=True, padding=True,
        max_length=max_tokens,
    )
    ids = batch["input_ids"].contiguous().cpu().numpy()
    mask = batch["attention_mask"].contiguous().cpu().numpy()
    capture_meta = _json(capture_path.with_suffix(".json"))
    tokens_sha = hashlib.sha256(ids.tobytes()).hexdigest()
    if capture_meta.get("token_ids_sha256") != tokens_sha:
        raise ValueError("text fixture and token limit do not match original capture")
    counts = mask.sum(axis=1).astype(np.int64)
    if int(counts.sum()) != token_count:
        raise ValueError("document token counts do not match captured states")
    document_ids = np.repeat(np.arange(len(lines)), counts)
    return document_ids, {i: int(counts[i]) for i in range(len(lines))}


def _as_floats(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().float()


def _prediction_metrics(
    teacher: torch.Tensor, prediction_latent: torch.Tensor, p: torch.Tensor,
) -> dict[str, float]:
    return comparison(teacher, prediction_latent @ p.T)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--calibration-uri", type=str, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--router-topk", type=Path, required=True)
    parser.add_argument("--texts-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--expert-id", type=int, choices=(77, 87, 246, 131),
                        required=True)
    parser.add_argument("--validation-document", type=int, default=None,
                        help="document index to reserve for validation; auto if omitted")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not 1 <= args.max_tokens <= 256
        or not 1 <= args.rank <= 64
        or args.seed < 0
    ):
        parser.error("output must be new; max-tokens 1..256, rank 1..64, seed >= 0")
    try:
        source_manifest_path = args.raw_dir / "raw_expert_manifest.json"
        manifest = _json(source_manifest_path)
        capacity = manifest.get("target_experts")
        if (
            manifest.get("source_repo") != REPO
            or manifest.get("source_revision") != REVISION
            or manifest.get("not_an_iq_checkpoint") is not True
            or type(capacity) is not int
            or capacity not in (64, 128, 256)
            or manifest.get("target_config_fingerprint")
            != canonical_glm53_config(routed_experts=capacity).fingerprint
        ):
            raise ValueError("invalid donor provenance or IQ configuration")
        rows = [item for item in manifest.get("layers", []) if item.get("stage") == 1]
        if len(rows) != 1 or rows[0].get("source_layer") != 3:
            raise ValueError("stage-1 source must be GLM layer 3")
        experts = rows[0]["exported_donor_ids"]
        if args.expert_id not in experts:
            raise ValueError("expert absent from original four-expert export")
        slot = experts.index(args.expert_id)
        activations = np.load(
            args.activations, allow_pickle=False, mmap_mode="r"
        )
        if (
            activations.ndim != 2
            or activations.shape[1] != 6144
            or activations.dtype != np.float32
            or not np.isfinite(activations).all()
        ):
            raise ValueError("invalid real GLM capture tensor")
        routed, routes_sha = read_verified_router_pairs(
            capture_path=args.activations, routing_path=args.router_topk,
            source_layer=3, samples_used=int(activations.shape[0]),
        )
        docs, doc_counts = _load_document_ids(
            args.texts_file, args.activations,
            max_tokens=args.max_tokens, token_count=int(activations.shape[0]),
        )
        selected = (routed == args.expert_id).any(axis=1)
        distribution = {
            i: int((selected & (docs == i)).sum()) for i in doc_counts
        }
        available = [
            i for i, c in distribution.items()
            if c >= 2 and int(selected.sum()) - c >= 8
        ]
        if args.validation_document is None:
            if not available:
                raise ValueError(
                    f"insufficient document-disjoint samples for expert "
                    f"{args.expert_id}: {distribution}; collect more tokens"
                )
            target = int(selected.sum()) / max(2, len(doc_counts))
            val_doc = min(available, key=lambda i: (abs(distribution[i] - target), i))
        else:
            val_doc = args.validation_document
            if val_doc not in available:
                raise ValueError(
                    f"requested validation document has insufficient route "
                    f"coverage for expert {args.expert_id}: {distribution}"
                )
        train_rows = np.where(selected & (docs != val_doc))[0]
        validation_rows = np.where(selected & (docs == val_doc))[0]

        import tempfile
        with tempfile.TemporaryDirectory(prefix="glm53-frozen-bridge-") as temp:
            root = Path(temp)
            _restore_stage_from_hf(args.calibration_uri, root, stage=1)
            basis_np, physical, verified = _calibration_map(root, 1, 3, capacity)
            basis = torch.from_numpy(np.asarray(
                basis_np, dtype=np.float32
            ).copy())
            _read_baseline(
                args.baseline_dir, _sha_file(source_manifest_path),
                experts, 1, verified,
            )
            projected = _load_baseline_expert(
                args.baseline_dir, slot, args.expert_id, physical,
                _sha_file(source_manifest_path), verified, 1,
                latent=basis.shape[1], intermediate=2048,
            )
        prefix = f"model.layers.3.mlp.experts.{args.expert_id}."
        donor = {
            kind: _read_donor(
                args.raw_dir, manifest, f"{prefix}{kind}_proj.weight",
                (2048, 6144) if kind != "down" else (6144, 2048),
            )
            for kind in ("gate", "up", "down")
        }
        hashes_before = {k: _sha_tensor(v) for k, v in donor.items()}
        torch.manual_seed(args.seed)
        bridge = FrozenGLMExpertBridge(
            donor, basis, rank=args.rank, seed=args.seed
        )
        real = torch.from_numpy(np.array(activations, copy=True))
        train_real = real[torch.from_numpy(train_rows)]
        val_real = real[torch.from_numpy(validation_rows)]
        with torch.no_grad():
            teacher_train = bridge.donor_forward(train_real)
            teacher_val = bridge.donor_forward(val_real)
            z_train = train_real @ basis
            z_val = val_real @ basis
            target_train = teacher_train @ basis
            target_val = teacher_val @ basis
            projected_val = (
                torch.nn.functional.linear(
                    torch.nn.functional.silu(
                        torch.nn.functional.linear(z_val, projected["gate"])
                    ) * torch.nn.functional.linear(z_val, projected["up"]),
                    projected["down"],
                )
            )
            initial_val = bridge(z_val)
            warm = _prediction_metrics(teacher_val, projected_val, basis)
            initial = _prediction_metrics(teacher_val, initial_val, basis)
        training = train_frozen_bridge(
            bridge, z_train, target_train, z_val, target_val,
            steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.lr, seed=args.seed,
        )
        with torch.no_grad():
            trained = _prediction_metrics(teacher_val, bridge(z_val), basis)
        hashes_after = {k: _sha_tensor(v) for k, v in
                        (("gate", bridge.gate), ("up", bridge.up), ("down", bridge.down))}
        if hashes_after != hashes_before:
            raise RuntimeError("frozen original GLM expert tensor mutated")
        # All model weights, WARM maps and captured inputs remain unmodified.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        adapted_file = args.output_dir / "adapter_only.safetensors"
        save_file(bridge.adapter_state(), str(adapted_file))
        report = {
            "schema_version": 1,
            "artifact_type": "glm53_immutable_donor_trainable_bridge_probe",
            "not_an_iq_checkpoint": True,
            "experimental_not_production": True,
            "source_repo": REPO,
            "source_revision": REVISION,
            "stage": 1,
            "source_layer": 3,
            "expert_id": args.expert_id,
            "recipient_moe_physical_layer": physical,
            "input_coordinate_source": "canonical_WARM_latent_control",
            "routed_samples": int(selected.sum()),
            "routed_counts_by_document": distribution,
            "validation_document_id": val_doc,
            "validation_samples": len(validation_rows),
            "train_document_disjoint": True,
            "independent_final_test": False,
            "note": ("Validation document is used for adapter checkpoint selection. "
                     "Independent final test data remains necessary."),
            "teacher_capture_sha256": _sha_file(args.activations),
            "router_topk_sha256": routes_sha,
            "text_fixture_sha256": _sha_file(args.texts_file),
            "canonical_calibration_sha256": verified,
            "donor_tensors_sha256": hashes_before,
            "adapter_only_sha256": _sha_file(adapted_file),
            "train_config": {
                "rank": args.rank, "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr, "seed": args.seed,
            },
            "training": training,
            "validation": {
                "warm_bf16_projected_baseline": warm,
                "frozen_donor_projection_only": initial,
                "frozen_donor_learned_bridge": trained,
            },
            "donor_bf16_tensors_unchanged": True,
        }
        report_file = args.output_dir / "report.json"
        report_file.write_text(json.dumps(
            report, indent=2, sort_keys=True
        ) + "\n", encoding="utf-8")
        print(json.dumps({
            "expert_id": args.expert_id,
            "routed_count": report["routed_samples"],
            "by_document": distribution,
            "train_count": training["training_samples"],
            "validation_count": training["evaluation_samples"],
            "original_warm_rmse": warm["relative_rmse"],
            "bridge_before_rmse": initial["relative_rmse"],
            "bridge_trained_rmse": trained["relative_rmse"],
            "original_bf16_weights_unchanged": True,
            "report": str(report_file),
        }, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Frozen expert bridge probe failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
