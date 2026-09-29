from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from iq_model import IQHybridConfig, require_mamba3_mimo_runtime
from iq_transfer.mamba3_direct import (
    Mamba3DonorConfig,
    _load_official_state_dict,
    validate_official_mamba3_mimo_15b_config,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CUDA parity check for direct Mamba-3 MIMO transplant shards"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--recipient-config", required=True)
    p.add_argument("--overlay", required=True)
    p.add_argument(
        "--source-layer",
        action="append",
        type=int,
        help="donor layer to verify; repeat as needed. Defaults to first/middle/last.",
    )
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--atol", type=float, default=5e-2)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be positive")

    device = torch.device("cuda")
    require_mamba3_mimo_runtime(device)

    checkpoint = Path(args.checkpoint)
    overlay = Path(args.overlay)
    donor_config = Mamba3DonorConfig.from_json(checkpoint / "config.json")
    validate_official_mamba3_mimo_15b_config(donor_config)
    recipient = IQHybridConfig.from_json(args.recipient_config)

    manifest = json.loads(
        (overlay / "mamba3_transplant.json").read_text(encoding="utf-8")
    )
    if manifest["recipient"]["config_fingerprint"] != recipient.fingerprint:
        raise RuntimeError("overlay recipient fingerprint does not match supplied config")

    placements = {
        int(item["source_layer"]): item
        for item in manifest["placements"]
    }
    requested = args.source_layer
    if not requested:
        requested = [
            0,
            donor_config.n_layer // 2,
            donor_config.n_layer - 1,
        ]
    if any(layer not in placements for layer in requested):
        raise RuntimeError("requested source layer is not present in overlay")

    state = _load_official_state_dict(checkpoint / "pytorch_model.bin")
    from mamba_ssm import Mamba3

    torch.manual_seed(0)
    reports = []
    for source_layer in requested:
        placement = placements[source_layer]
        source = Mamba3(
            d_model=donor_config.d_model,
            d_state=donor_config.d_state,
            expand=donor_config.expand,
            headdim=donor_config.headdim,
            ngroups=donor_config.ngroups,
            rope_fraction=donor_config.rope_fraction,
            is_outproj_norm=donor_config.is_outproj_norm,
            is_mimo=True,
            mimo_rank=donor_config.mimo_rank,
            chunk_size=donor_config.chunk_size,
            layer_idx=source_layer,
            n_layer=donor_config.n_layer,
            device=device,
            dtype=torch.bfloat16,
        )
        sp = f"backbone.layers.{source_layer}.mixer."
        source_sd = {
            key[len(sp):]: value.to(device=device)
            for key, value in state.items()
            if key.startswith(sp)
        }
        source.load_state_dict(source_sd, strict=True)
        source.eval()

        target = Mamba3(
            d_model=recipient.mamba3.d_model,
            d_state=recipient.mamba3.d_state,
            expand=recipient.mamba3.expand,
            headdim=recipient.mamba3.headdim,
            ngroups=1,
            rope_fraction=recipient.mamba3.rope_fraction,
            is_outproj_norm=recipient.mamba3.outproj_norm,
            is_mimo=True,
            mimo_rank=recipient.mamba3.mimo_rank,
            chunk_size=recipient.mamba3.chunk_size,
            layer_idx=int(placement["target_mamba_ordinal"]),
            n_layer=recipient.mamba3.num_layers,
            device=device,
            dtype=torch.bfloat16,
        )
        shard = load_file(str(overlay / placement["shard"]), device="cpu")
        tp = f"layers.{int(placement['target_physical_layer'])}.mamba.core."
        target_sd = {
            key[len(tp):]: value.to(device=device)
            for key, value in shard.items()
            if key.startswith(tp)
        }
        target.load_state_dict(target_sd, strict=True)
        target.eval()

        x_source = torch.randn(
            1,
            args.sequence_length,
            donor_config.d_model,
            device=device,
            dtype=torch.bfloat16,
        )
        x_target = torch.zeros(
            1,
            args.sequence_length,
            recipient.mamba3.d_model,
            device=device,
            dtype=torch.bfloat16,
        )
        x_target[..., : donor_config.d_model].copy_(x_source)

        with torch.inference_mode():
            y_source = source(x_source)
            y_target = target(x_target)

        donor_view = y_target[..., : donor_config.d_model]
        extra_view = y_target[..., donor_config.d_model :]
        max_abs = float(
            (donor_view.float() - y_source.float()).abs().max().item()
        )
        extra_max_abs = float(extra_view.float().abs().max().item())
        parity = torch.allclose(
            donor_view.float(),
            y_source.float(),
            rtol=args.rtol,
            atol=args.atol,
        )
        neutral = extra_max_abs == 0.0
        reports.append(
            {
                "source_layer": source_layer,
                "target_physical_layer": int(placement["target_physical_layer"]),
                "max_abs_error": max_abs,
                "extra_subspace_max_abs": extra_max_abs,
                "parity": bool(parity),
                "extra_subspace_neutral": bool(neutral),
            }
        )
        del source, target, x_source, x_target, y_source, y_target
        torch.cuda.empty_cache()

    print(json.dumps({"layers": reports}, indent=2, sort_keys=True))
    if not all(r["parity"] and r["extra_subspace_neutral"] for r in reports):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
