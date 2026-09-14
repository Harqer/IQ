from __future__ import annotations

import argparse

from transformers import AutoConfig

from iq_model import IQArchitectureConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IQ hybrid-v1 dimensions against Phi-4-mini without loading weights"
    )
    parser.add_argument("--donor", default="microsoft/Phi-4-mini-instruct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donor = AutoConfig.from_pretrained(args.donor, trust_remote_code=True)
    cfg = IQArchitectureConfig()
    cfg.validate_teacher_depth(donor.num_hidden_layers)

    head_dim = donor.hidden_size // donor.num_attention_heads
    gdn_key_width = round(donor.hidden_size * cfg.gdn_key_width_ratio)
    if gdn_key_width % cfg.gdn_head_dim != 0:
        raise RuntimeError(
            f"GDN key width {gdn_key_width} is not divisible by head dim {cfg.gdn_head_dim}"
        )
    gdn_heads = gdn_key_width // cfg.gdn_head_dim

    if donor.hidden_size % donor.num_attention_heads != 0:
        raise RuntimeError("donor hidden size is not divisible by attention heads")
    if donor.num_attention_heads % donor.num_key_value_heads != 0:
        raise RuntimeError("donor attention heads are not divisible by KV heads")

    print(f"donor={args.donor}")
    print("profile=hybrid_v1")
    print(f"hidden_size={donor.hidden_size}")
    print(f"intermediate_size={donor.intermediate_size}")
    print(f"teacher_layers={donor.num_hidden_layers}")
    print(f"physical_layers={cfg.physical_layers}")
    print(f"effective_depth={cfg.effective_depth}")
    print(f"core_schedule={cfg.core_mixer_schedule}")
    print(f"gdn_key_width={gdn_key_width}")
    print(f"gdn_head_dim={cfg.gdn_head_dim}")
    print(f"gdn_heads={gdn_heads}")
    print(f"nsa_query_heads={donor.num_attention_heads}")
    print(f"nsa_kv_heads={donor.num_key_value_heads}")
    print(f"nsa_head_dim={head_dim}")
    print(f"nsa_block_size={cfg.nsa_block_size}")
    print(f"nsa_block_count={cfg.nsa_block_count}")
    print(f"nsa_window_size={cfg.nsa_window_size}")
    print(f"position_strategy={cfg.attention_position_strategy}")
    print(f"ffn_kind={cfg.ffn_kind}")

    assert cfg.physical_layers == 16
    assert cfg.effective_depth == 32
    assert cfg.core_mixer_schedule.count("gated_deltanet") == 5
    assert cfg.core_mixer_schedule.count("nsa") == 3
    assert donor.hidden_size == donor.num_attention_heads * head_dim
    assert gdn_heads > 0

    print("hybrid-v1 config validation: PASS")


if __name__ == "__main__":
    main()
