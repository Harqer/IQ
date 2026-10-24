#!/usr/bin/env python3
"""Bounded IQ behavior correction from genuine GLM text examples.

Does NOT load GLM logits, copy extra weights, or turn IQ into GLM. Produces a
small, separately versioned delta for validation; never mutates source bucket.
Only run on an explicitly provisioned CUDA machine with checkpoint + data.
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

from iq_model import HybridLayerType
from iq_training.behavior_transfer import (SUPPORTED_GLM_TEACHERS, causal_response_loss)
from iq_training.mohawk import freeze_for_glm53_stage2
from iq_transfer import load_complete_iq_checkpoint


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_dataset(data: Path, tokenizer_dir: Path) -> dict:
    meta = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or meta.get("purpose") != "iq_response_supervision":
        raise ValueError("not a supported IQ behavior-transfer artifact")
    if meta.get("teacher_source_claim") not in SUPPORTED_GLM_TEACHERS:
        raise ValueError("teacher model claim does not identify GLM-5.3")
    if meta.get("teacher_attribution_verified_by_api") is not False:
        raise ValueError("unexpected teacher authenticity field")
    if digest(tokenizer_dir / "tokenizer.json") != meta.get("tokenizer_json_sha256"):
        raise ValueError("IQ tokenizer.json differs from dataset")
    if digest(tokenizer_dir / "tokenizer_config.json") != meta.get("tokenizer_config_sha256"):
        raise ValueError("IQ tokenizer_config.json differs from dataset")
    train_digests = set(meta["split"]["train"]["content_digests"])
    val_digests = set(meta["split"]["validation"]["content_digests"])
    if not train_digests or not val_digests or train_digests & val_digests:
        raise ValueError("invalid or leaking train-validation split")
    for split in ("train", "validation"):
        path = data / f"{split}.safetensors"
        if digest(path) != meta["split"][split]["sha256"]:
            raise ValueError(f"dataset hash mismatch: {split}")
    return meta


def examples_from_split(data: Path, split: str, count: int):
    tensors = load_file(str(data / f"{split}.safetensors"), device="cpu")
    required = {"input_ids", "labels", "attention_mask"}
    expected = {
        f"{i:06d}/{name}" for i in range(count) for name in required
    }
    if set(tensors) != expected:
        raise ValueError(f"{split} tensor schema/count mismatch")
    examples = []
    for i in range(count):
        fields = {name: tensors[f"{i:06d}/{name}"] for name in required}
        n = fields["input_ids"].numel()
        if n < 2 or any(value.shape != (n,) for value in fields.values()):
            raise ValueError("invalid behavior token shapes")
        if fields["labels"].dtype != torch.long or fields["input_ids"].dtype != torch.long:
            raise ValueError("expected int64 token IDs and labels")
        if fields["attention_mask"].dtype != torch.bool or not bool(fields["attention_mask"].all()):
            raise ValueError("expected non-padded single-example sequences")
        if not bool((fields["labels"][1:] != -100).any()):
            raise ValueError("example contains no supervised continuation")
        examples.append(fields)
    return examples


def loss_for_example(model, example: dict, device: torch.device):
    ids = example["input_ids"].to(device).unsqueeze(0)
    mask = example["attention_mask"].to(device).unsqueeze(0)
    labels = example["labels"].to(device).unsqueeze(0)
    output = model(ids, attention_mask=mask, labels=None)
    return causal_response_loss(output.logits, labels)


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()
    values = [float(loss_for_example(model, item, device)) for item in examples]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite behavior validation loss")
    return sum(values) / len(values)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dry-run", action="store_true", help="validate dataset and checkpoint metadata without GPU")
    args = p.parse_args()
    if not 1 <= args.steps <= 1000 or not 0 < args.lr <= 1e-3:
        p.error("steps must be 1..1000, LR must be (0,0.001]")
    if args.out_dir.exists():
        p.error("out-dir exists; source and training outputs must never be overwritten")

    try:
        meta = inspect_dataset(args.data, args.tokenizer_dir)
        index_path = args.checkpoint / "model.safetensors.index.json"
        config = json.loads((args.checkpoint / "iq_config.json").read_text(encoding="utf-8"))
        if config.get("model", {}).get("vocab_size") != 128256:
            raise ValueError("IQ checkpoint vocabulary mismatch")
        provenance = json.loads(
            (args.checkpoint / "transplant_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            provenance.get("capability_source", {}).get("revision")
            != meta["teacher_revision_claim"]
        ):
            raise ValueError(
                "teacher revision claim differs from GLM donor used in original IQ weights"
            )
        parent_index_sha = digest(index_path)
        print("Preflight passed: provenance, tokenizer and data file hashes verified.")
        print("Teacher attribution is a claim from input records, not API-authenticated.")
        print(f"Train={meta['split']['train']['count']}; "
              f"validation={meta['split']['validation']['count']}")
        if args.dry_run:
            print("Dry-run ended; no model loaded, no GPU used, no transfer claimed.")
            return 0
        if not torch.cuda.is_available():
            raise ValueError("actual 17B IQ training requires explicitly provisioned CUDA")
        device = torch.device("cuda:0")
        train = examples_from_split(args.data, "train", meta["split"]["train"]["count"])
        val = examples_from_split(args.data, "validation", meta["split"]["validation"]["count"])
        model = load_complete_iq_checkpoint(
            args.checkpoint, device=device, dtype=torch.bfloat16
        )
        schedule = model.config.schedule.layers
        anchors = [
            i for i in range(1, len(schedule))
            if schedule[i] is HybridLayerType.MOE
            and schedule[i - 1] in {HybridLayerType.CSA, HybridLayerType.HCA}
        ]
        if not anchors:
            raise ValueError("recipient has no GLM context+MoE stage")
        target = anchors[-1]
        selected = freeze_for_glm53_stage2(
            model, target_physical_layer=target, train_preceding_mamba=False
        )
        model.norm.requires_grad_(True)
        optimizer_params = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr)
        initial_val_loss = evaluate(model, val, device)
        print(f"initial validation response loss: {initial_val_loss:.6f}", flush=True)
        trained_steps = []
        for step in range(args.steps):
            model.train()
            example = train[step % len(train)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_example(model, example, device)
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("non-finite gradient norm")
            optimizer.step()
            value = float(loss.detach())
            trained_steps.append(value)
            print(f"step {step + 1}/{args.steps} train_response_ce={value:.6f}", flush=True)
        final_val_loss = evaluate(model, val, device)
        updated = {
            name: param.detach().cpu().contiguous()
            for name, param in model.named_parameters() if param.requires_grad
        }
        if not updated:
            raise ValueError("no updated IQ tensors selected")
        args.out_dir.mkdir(parents=True)
        delta_path = args.out_dir / "behavior_delta.safetensors"
        save_file(updated, str(delta_path))
        result = {
            "schema_version": 1,
            "status": "candidate_not_promoted",
            "language_quality_verified": False,
            "trainable_layers": list(selected),
            "delta_tensors": len(updated),
            "delta_sha256": digest(delta_path),
            "parent_checkpoint_index_sha256": parent_index_sha,
            "data_manifest_sha256": digest(args.data / "manifest.json"),
            "teacher_model_claim": meta["teacher_source_claim"],
            "teacher_revision_claim": meta["teacher_revision_claim"],
            "teacher_attribution_verified_by_api": False,
            "steps": args.steps,
            "lr": args.lr,
            "initial_validation_ce": initial_val_loss,
            "final_validation_ce": final_val_loss,
            "train_loss_history": trained_steps,
        }
        (args.out_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Saved candidate delta: {delta_path}; source checkpoint untouched.")
        print(f"validation CE: {initial_val_loss:.4f} -> {final_val_loss:.4f}")
        print("Language behavior NOT verified; do not promote without held-out QA.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Transfer stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
