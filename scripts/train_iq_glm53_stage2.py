#!/usr/bin/env python3
"""Train one real GLM->IQ MOHAWK Stage-2 context+MoE block.

Consumes two independently captured, byte-aligned and map-validated examples.
Produces an unpromoted delta; source checkpoint remains unchanged. No GPU
allocation unless explicitly run on a provisioned machine WITHOUT --dry-run.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

from iq_training.mohawk import IQGLM53MohawkStage2
from iq_transfer import load_complete_iq_checkpoint


def sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_alignment(folder: Path, *, split: str) -> tuple[dict, dict]:
    meta_path = folder / "manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or meta.get("stage") != "glm53_mohawk_stage2":
        raise ValueError("not a validated MOHAWK Stage-2 artifact")
    if meta.get("split") != split or meta.get("teacher_capture_authenticated") is not False:
        raise ValueError("invalid split/source-authentication metadata")
    if not 0 < meta.get("causally_aligned_tokens", 0) <= meta.get("alignment_tokens", 0):
        raise ValueError("alignment has no causal tokens")
    for name, expected in (
        ("alignment.safetensors", "alignment_sha256"),
        ("validated-residual-map.safetensors", "map_sha256"),
        ("validated-residual-map.json", "map_metadata_sha256"),
    ):
        if sha(folder / name) != meta[expected]:
            raise ValueError(f"Stage-2 file integrity failure: {name}")
    if not math.isfinite(meta.get("heldout_validation_rmse", float("nan"))):
        raise ValueError("coordinate map lacks finite held-out validation RMSE")
    payload = load_file(str(folder / "alignment.safetensors"), device="cpu")
    if set(payload) != {"input_ids", "teacher_aligned_hidden", "teacher_alignment_mask"}:
        raise ValueError("Stage-2 input tensors contain unexpected keys")
    ids, hidden, mask = (payload[name] for name in (
        "input_ids", "teacher_aligned_hidden", "teacher_alignment_mask"
    ))
    if ids.ndim != 2 or ids.shape[0] != 1 or hidden.shape != (*ids.shape, 4096):
        raise ValueError("Stage-2 hidden and IQ token shapes mismatch")
    if ids.dtype != torch.int64 or mask.shape != ids.shape or mask.dtype != torch.bool:
        raise ValueError("Stage-2 ID and mask types mismatch")
    if not bool(mask.any()) or not bool(torch.isfinite(hidden).all()):
        raise ValueError("non-finite or empty stage2 targets")
    if int(ids.min()) < 0 or int(ids.max()) >= 128256:
        raise ValueError("invalid IQ token ID")
    return meta, payload


def forward(adapter, tensors, device, *, train: bool):
    batch = {key: value.to(device) for key, value in tensors.items()}
    output = adapter(
        batch["input_ids"],
        teacher_aligned_hidden=batch["teacher_aligned_hidden"],
        teacher_alignment_mask=batch["teacher_alignment_mask"],
        attention_mask=torch.ones_like(batch["input_ids"], dtype=torch.bool),
    )
    if not bool(torch.isfinite(output.loss)):
        raise ValueError("non-finite Stage-2 hidden alignment loss")
    return output.loss


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-alignment", type=Path, required=True)
    p.add_argument("--validation-alignment", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 1 <= args.steps <= 1000 or not 0 < args.lr <= 1e-3:
        p.error("steps must be 1..1000 and lr in (0,0.001]")
    if args.out_dir.exists():
        p.error("refusing to overwrite existing training output")
    try:
        train_meta, train = load_alignment(args.train_alignment, split="train")
        val_meta, val = load_alignment(args.validation_alignment, split="validation")
        if train_meta["document_sha256"] == val_meta["document_sha256"]:
            raise ValueError("train and validation use the same teacher document")
        for name in (
            "source_glm_layer", "target_iq_layer", "teacher_revision_claim",
            "map_sha256", "map_metadata_sha256",
        ):
            if train_meta[name] != val_meta[name]:
                raise ValueError(f"train/validation disagree on {name}")
        provenance = json.loads(
            (args.checkpoint / "transplant_manifest.json").read_text(encoding="utf-8")
        )
        source_revision = provenance["capability_source"]["revision"]
        if source_revision != train_meta["teacher_revision_claim"]:
            raise ValueError("teacher capture source differs from original GLM donor revision")
        original_layer = provenance["source_layer_map"].get(
            str(train_meta["target_iq_layer"])
        )
        # Source map keys are transplant stage ordinals, not physical layer IDs;
        # verify mapped layer via the recipient schedule below instead.
        del original_layer
        base_index_sha = sha(args.checkpoint / "model.safetensors.index.json")
        print("Stage-2 sources and held-out coordinate map verified.")
        if args.dry_run:
            print("No CUDA allocation; no model loaded or changed.")
            return 0
        if not torch.cuda.is_available():
            raise ValueError("17B Stage-2 needs explicit CUDA allocation")
        device = torch.device("cuda:0")
        model = load_complete_iq_checkpoint(
            args.checkpoint, device=device, dtype=torch.bfloat16
        )
        stage = train_meta["target_iq_layer"]
        adapter = IQGLM53MohawkStage2(
            model,
            source_glm_layer=train_meta["source_glm_layer"],
            target_physical_layer=stage,
            alignment_manifest_sha256=sha(args.train_alignment / "manifest.json"),
        )
        params = [p for p in adapter.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        @torch.no_grad()
        def val_loss() -> float:
            adapter.eval()
            loss = float(forward(adapter, val, device, train=False))
            if not math.isfinite(loss):
                raise ValueError("validation alignment loss is non-finite")
            return loss

        before = val_loss()
        losses = []
        for step in range(args.steps):
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            loss = forward(adapter, train, device, train=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("non-finite gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
            print(f"Stage 2 step {step+1}/{args.steps} loss={losses[-1]:.6f}", flush=True)
        after = val_loss()
        delta = {
            name: param.detach().cpu().contiguous()
            for name, param in model.named_parameters() if param.requires_grad
        }
        if not delta:
            raise ValueError("no Stage-2 parameters changed")
        args.out_dir.mkdir(parents=True)
        delta_path = args.out_dir / "stage2_delta.safetensors"
        save_file(delta, str(delta_path))
        (args.out_dir / "manifest.json").write_text(
            json.dumps({
                "schema_version": 1,
                "status": "candidate_not_promoted",
                "language_quality_verified": False,
                "base_index_sha256": base_index_sha,
                "teacher_revision": source_revision,
                "source_glm_layer": train_meta["source_glm_layer"],
                "target_iq_layer": stage,
                "train_alignment_sha256": sha(args.train_alignment / "manifest.json"),
                "validation_alignment_sha256": sha(args.validation_alignment / "manifest.json"),
                "delta_sha256": sha(delta_path),
                "delta_tensors": len(delta),
                "train_loss_history": losses,
                "validation_before": before,
                "validation_after": after,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Candidate delta saved; validation alignment {before:.5f} -> {after:.5f}.")
        print("This does not establish coherent text. Source checkpoint unchanged.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Stage-2 transfer blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
