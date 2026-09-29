from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from iq_model import Mamba3MIMOConfig, require_mamba3_mimo_runtime
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
    target_config = Mamba3MIMOConfig.production_4096x32()

    manifest = json.loads(
        (overlay / "mamba3_transplant.json").read_text(encoding="utf-8")
    )
    manifest_target = manifest.get("target", {}).get("mamba3")
    if not isinstance(manifest_target, dict):
        raise RuntimeError("overlay is missing target Mamba config")
    if manifest_target != target_config.__dict__:
        raise RuntimeError("overlay target Mamba config does not match frozen production target")

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
        target_ordinal = int(placement["target_mamba_ordinal"])
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
            d_model=target_config.d_model,
            d_state=target_config.d_state,
            expand=target_config.expand,
            headdim=target_config.headdim,
            ngroups=1,
            rope_fraction=target_config.rope_fraction,
            is_outproj_norm=target_config.outproj_norm,
            is_mimo=True,
            mimo_rank=target_config.mimo_rank,
            chunk_size=target_config.chunk_size,
            layer_idx=target_ordinal,
            n_layer=target_config.num_layers,
            device=device,
            dtype=torch.bfloat16,
        )
        shard = load_file(str(overlay / placement["shard"]), device="cpu")
        tp = f"mamba_layers.{target_ordinal}.core."
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
        factor = target_config.d_model // donor_config.d_model
        x_target = x_source.repeat((1, 1, factor))

        source_norm_weight = state[
            f"backbone.layers.{source_layer}.norm.weight"
        ].to(device=device)
        target_norm_weight = shard[
            f"mamba_layers.{target_ordinal}.norm.weight"
        ].to(device=device)

        def rms_norm(
            value: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            normalized = value.float() * torch.rsqrt(
                value.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5
            )
            return (normalized * weight.float()).to(value.dtype)

        source_input = rms_norm(x_source, source_norm_weight)
        target_input = rms_norm(x_target, target_norm_weight)

        with torch.inference_mode():
            y_source = source(source_input)
            y_target = target(target_input)

        expected_mixed = y_source.repeat((1, 1, factor))
        source_residual = x_source + y_source
        target_residual = x_target + y_target
        expected_residual = source_residual.repeat((1, 1, factor))
        mixed_max_abs = float(
            (y_target.float() - expected_mixed.float()).abs().max().item()
        )
        residual_max_abs = float(
            (target_residual.float() - expected_residual.float()).abs().max().item()
        )
        normalized_max_abs = float(
            (
                target_input.float()
                - source_input.repeat((1, 1, factor)).float()
            ).abs().max().item()
        )
        parity = (
            torch.allclose(
                target_input.float(),
                source_input.repeat((1, 1, factor)).float(),
                rtol=args.rtol,
                atol=args.atol,
            )
            and torch.allclose(
                y_target.float(),
                expected_mixed.float(),
                rtol=args.rtol,
                atol=args.atol,
            )
            and torch.allclose(
                target_residual.float(),
                expected_residual.float(),
                rtol=args.rtol,
                atol=args.atol,
            )
        )
        reports.append(
            {
                "source_layer": source_layer,
                "target_mamba_ordinal": target_ordinal,
                "norm_max_abs_error": normalized_max_abs,
                "mixer_max_abs_error": mixed_max_abs,
                "residual_max_abs_error": residual_max_abs,
                "replication_factor": factor,
                "parity": bool(parity),
            }
        )
        del source, target, x_source, x_target, y_source, y_target
        torch.cuda.empty_cache()

    print(json.dumps({"layers": reports}, indent=2, sort_keys=True))
    if not all(r["parity"] for r in reports):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
