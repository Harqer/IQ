from __future__ import annotations

import argparse
import importlib.util

import torch
from transformers import AutoConfig

from iq_model import IQArchitectureConfig, IQRecurrentPhiModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IQ Mamba-v2 geometry against the Phi-4-mini donor config without loading donor weights"
    )
    parser.add_argument("--donor", default="microsoft/Phi-4-mini-instruct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donor = AutoConfig.from_pretrained(args.donor, trust_remote_code=True)
    donor._attn_implementation = "eager"
    iq = IQArchitectureConfig()
    iq.validate_teacher_depth(donor.num_hidden_layers)

    inner = donor.hidden_size * iq.mamba3_expand
    if inner % iq.mamba3_head_dim != 0:
        raise RuntimeError("real donor width is incompatible with configured Mamba-3 head geometry")

    print(f"donor={args.donor}")
    print(f"hidden_size={donor.hidden_size}")
    print(f"teacher_layers={donor.num_hidden_layers}")
    print(f"iq_physical_layers={iq.physical_layers}")
    print(f"iq_effective_depth={iq.effective_depth}")
    print(f"core_schedule={iq.core_mixer_schedule}")
    print(f"mamba3_state_size={iq.mamba3_state_size}")
    print(f"mamba3_inner_size={inner}")
    print(f"mamba3_heads={inner // iq.mamba3_head_dim}")
    print(f"mamba3_mimo_rank={iq.mamba3_mimo_rank}")
    print(f"mamba3_chunk_size={iq.mamba3_chunk_size}")
    print(f"jepa_latent_predictor={iq.use_latent_predictor}")

    if importlib.util.find_spec("mamba_ssm") is None or importlib.util.find_spec("fla") is None:
        raise RuntimeError("install requirements-mamba-v2.txt before constructing the hybrid model")

    with torch.device("meta"):
        model = IQRecurrentPhiModel(donor, iq)

    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"iq_parameters={params:,}")

    assert model.physical_depth == 16
    assert model.effective_depth == donor.num_hidden_layers
    assert model.reasoning_core.mixer_schedule == iq.core_mixer_schedule
    assert model.latent_predictor is not None
    assert iq.core_mixer_schedule.count("mamba3_mimo") == 6
    assert iq.core_mixer_schedule.count("nsa") == 2

    print("mamba-v2 architecture validation: PASS")


if __name__ == "__main__":
    main()
