from __future__ import annotations

import argparse
from pathlib import Path

import torch

from iq_model import IQHybridConfig, IQHybridForCausalLM, MTPConfig
from iq_training import (
    IQPretrainingModel,
    PretrainingObjectiveConfig,
)
from iq_transfer import load_token_batches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the serialized heterogeneous IQ hybrid on H200."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--mtp-depth", type=int, default=2)
    parser.add_argument("--mtp-weight", type=float, required=True)
    parser.add_argument("--load-balance-weight", type=float, required=True)
    parser.add_argument("--router-z-weight", type=float, required=True)
    return parser


def _require_h200() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("hybrid verification requires CUDA")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if "H200" not in name:
        raise SystemExit(
            f"hybrid verification must run on H200; found {name!r}"
        )
    return device


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in batch.items()
    }


def _document_isolation_check(
    model: IQHybridForCausalLM,
    batch: dict[str, torch.Tensor],
) -> None:
    docs = batch.get("document_ids")
    if docs is None:
        raise AssertionError(
            "H200 hybrid gate requires document_ids to verify packed-state isolation"
        )
    mask = batch.get("attention_mask")
    valid = (
        mask.to(dtype=torch.bool)
        if mask is not None
        else torch.ones_like(docs, dtype=torch.bool)
    )

    row = 0
    valid_docs = docs[row][valid[row]]
    if valid_docs.numel() == 0:
        raise AssertionError("verification batch has no valid tokens")
    unique = torch.unique_consecutive(valid_docs)
    if unique.numel() < 2:
        raise AssertionError(
            "H200 hybrid gate requires at least two packed documents in the first row"
        )

    first_doc = unique[0]
    second_doc = unique[1]
    first_mask = valid[row] & (docs[row] == first_doc)
    second_mask = valid[row] & (docs[row] == second_doc)
    if not bool(first_mask.any()) or not bool(second_mask.any()):
        raise AssertionError("packed-document masks are empty")

    original = batch["input_ids"]
    modified = original.clone()
    vocab_size = model.model_config.vocab_size
    modified[row, first_mask] = (
        modified[row, first_mask] + 1
    ) % vocab_size

    model.eval()
    with torch.no_grad():
        baseline = model(
            original,
            position_ids=batch.get("position_ids"),
            attention_mask=mask,
            document_ids=docs,
        ).logits
        changed = model(
            modified,
            position_ids=batch.get("position_ids"),
            attention_mask=mask,
            document_ids=docs,
        ).logits

    torch.testing.assert_close(
        baseline[row, second_mask].float(),
        changed[row, second_mask].float(),
        rtol=1e-3,
        atol=1e-3,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.mtp_depth <= 0:
        raise SystemExit("--mtp-depth must be positive")
    device = _require_h200()

    config = IQHybridConfig.from_json(args.config)
    artifact = load_token_batches(args.batches)
    batch = _to_device(dict(artifact.batches[0]), device)
    batch["labels"] = batch["input_ids"]

    model = IQHybridForCausalLM(
        config,
        dtype=torch.bfloat16,
        device=device,
    )
    training_model = IQPretrainingModel(
        model,
        PretrainingObjectiveConfig(
            mtp_loss_weight=args.mtp_weight,
            moe_load_balance_loss_weight=args.load_balance_weight,
            moe_router_z_loss_weight=args.router_z_weight,
        ),
        mtp_config=MTPConfig(
            num_prediction_layers=args.mtp_depth,
        ),
    ).to(device)

    training_model.train()
    output = training_model(**batch)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise AssertionError("hybrid pretraining loss is missing/non-finite")
    if output.ntp_loss is None or not bool(torch.isfinite(output.ntp_loss)):
        raise AssertionError("next-token loss is missing/non-finite")
    if output.mtp_loss is None or not bool(torch.isfinite(output.mtp_loss)):
        raise AssertionError("MTP loss is missing/non-finite")
    if (
        output.moe_load_balance_loss is None
        or not bool(torch.isfinite(output.moe_load_balance_loss))
    ):
        raise AssertionError("MoE load-balance loss is missing/non-finite")
    if (
        output.moe_router_z_loss is None
        or not bool(torch.isfinite(output.moe_router_z_loss))
    ):
        raise AssertionError("MoE router-z loss is missing/non-finite")

    output.loss.backward()
    missing_required: list[str] = []
    nonfinite: list[str] = []
    routed_expert_gradients = 0
    for name, parameter in training_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            if ".moe.experts." in name:
                continue
            missing_required.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if ".moe.experts." in name:
            routed_expert_gradients += 1
    if routed_expert_gradients == 0:
        missing_required.append("at least one routed MoE expert gradient")
    if missing_required or nonfinite:
        raise AssertionError(
            "hybrid gradient failure: "
            f"missing_required={missing_required}, nonfinite={nonfinite}"
        )

    _document_isolation_check(model, batch)

    print(
        "IQ heterogeneous hybrid H200 gate passed:",
        {
            "config": str(Path(args.config)),
            "config_fingerprint": config.fingerprint,
            "schedule_fingerprint": config.schedule.fingerprint,
            "device": torch.cuda.get_device_name(device),
            "loss": float(output.loss.detach()),
            "ntp_loss": float(output.ntp_loss.detach()),
            "mtp_loss": float(output.mtp_loss.detach()),
            "moe_load_balance_loss": float(
                output.moe_load_balance_loss.detach()
            ),
            "moe_router_z_loss": float(
                output.moe_router_z_loss.detach()
            ),
        },
    )


if __name__ == "__main__":
    main()
