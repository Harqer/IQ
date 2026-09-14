from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig

from iq_model import DenseToRecurrentLayout, IQArchitectureConfig, IQRecurrentPhiModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IQ v0 Phi-control topology against Phi-4-mini without loading weights")
    parser.add_argument("--donor", default="microsoft/Phi-4-mini-instruct")
    return parser.parse_args()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    args = parse_args()
    donor = AutoConfig.from_pretrained(args.donor, trust_remote_code=True)
    donor._attn_implementation = "eager"
    # Explicitly pin the old validator to the Phi-only control. On hybrid-v1 the
    # default architecture is Gated DeltaNet + NSA and must not redefine v0.
    iq_cfg = IQArchitectureConfig.phi_control()
    iq_cfg.validate_teacher_depth(donor.num_hidden_layers)

    with torch.device("meta"):
        model = IQRecurrentPhiModel(donor, iq_cfg)

    layout = DenseToRecurrentLayout.from_config(iq_cfg)
    first_core = model.reasoning_core.blocks[0]

    print(f"donor={args.donor}")
    print("profile=phi_control")
    print(f"vocab_size={donor.vocab_size}")
    print(f"hidden_size={donor.hidden_size}")
    print(f"intermediate_size={donor.intermediate_size}")
    print(f"attention_heads={donor.num_attention_heads}")
    print(f"kv_heads={donor.num_key_value_heads}")
    print(f"teacher_layers={donor.num_hidden_layers}")
    print(f"iq_physical_layers={model.physical_depth}")
    print(f"iq_effective_depth={model.effective_depth}")
    print(f"iq_parameters={parameter_count(model):,}")
    print(f"core0_teacher_layers={layout.teacher_layers_for_core_block(0)}")
    print(f"core7_teacher_layers={layout.teacher_layers_for_core_block(iq_cfg.recurrent_layers - 1)}")
    print(f"qkv_shape={tuple(first_core.mixer.qkv_proj.weight.shape)}")
    print(f"ffn_gate_up_shape={tuple(first_core.feed_forward.gate_up_proj.weight.shape)}")
    print(f"ffn_down_shape={tuple(first_core.feed_forward.down_proj.weight.shape)}")

    assert model.effective_depth == donor.num_hidden_layers
    assert first_core.feed_forward.gate_up_proj.weight.shape == (
        2 * donor.intermediate_size,
        donor.hidden_size,
    )
    assert first_core.feed_forward.down_proj.weight.shape == (
        donor.hidden_size,
        donor.intermediate_size,
    )
    assert layout.teacher_layers_for_core_block(0)[0] == iq_cfg.prelude_layers
    assert layout.teacher_layers_for_core_block(iq_cfg.recurrent_layers - 1)[-1] == (
        iq_cfg.prelude_layers + iq_cfg.recurrent_layers * iq_cfg.recurrent_passes - 1
    )

    print("architecture v0 control validation: PASS")


if __name__ == "__main__":
    main()
