from __future__ import annotations

from dataclasses import dataclass


VALID_MIXERS = {
    "phi_gqa",
    "gated_deltanet",
    "nsa",
    "path_attention",
    "latent_nsa",
}


@dataclass(frozen=True)
class IQArchitectureConfig:
    """Topology for IQ hybrid v1.

    The recurrent middle stack is heterogeneous: cheap recurrent sequence mixers
    carry most tokens, while sparse attention anchors periodically recover exact
    long-range context.  Depth recurrence remains a separate axis and reuses the
    same physical stack across reasoning passes.

    `latent_nsa` is a target architecture identifier, not a silent alias for NSA.
    It remains unavailable until the MLA/GLA + NSA composition is implemented and
    validated against the published latent-NSA formulation.
    """

    prelude_layers: int = 4
    recurrent_layers: int = 8
    recurrent_passes: int = 3
    coda_layers: int = 4

    # Hybrid recurrent-core schedule agreed for the first experiment:
    # 5 recurrent linear mixers + 3 sparse attention anchors.
    core_mixer_schedule: tuple[str, ...] = (
        "gated_deltanet",
        "gated_deltanet",
        "nsa",
        "gated_deltanet",
        "gated_deltanet",
        "nsa",
        "gated_deltanet",
        "nsa",
    )

    # Keep the unique input/output blocks Phi-compatible during the first
    # transition so recurrence/mixer changes can be measured independently.
    prelude_mixer: str = "phi_gqa"
    coda_mixer: str = "phi_gqa"

    # Recurrent-depth state.
    use_pass_embeddings: bool = True
    recurrent_delta_scale: float = 1.0

    # Gated DeltaNet settings.  For Phi-4-mini, 0.75 * 3072 / 128 = 18 heads,
    # matching the parameter allocation recommended by the FLA implementation.
    gdn_key_width_ratio: float = 0.75
    gdn_head_dim: int = 128
    gdn_expand_v: float = 2.0
    gdn_use_short_conv: bool = True
    gdn_conv_size: int = 4

    # Native Sparse Attention settings.
    nsa_block_size: int = 64
    nsa_block_count: int = 16
    nsa_window_size: int = 512

    # Position strategy for sparse/global attention. `rope` is the validated
    # execution baseline. `path` is exposed through a dedicated PaTH mixer rather
    # than being falsely treated as a drop-in replacement inside NSA.
    attention_position_strategy: str = "rope"

    # FFN remains dense SwiGLU for the first hybrid-mixer transition. Shared/routed
    # MoE is a separate controlled change after the hybrid core is stable.
    ffn_kind: str = "dense_swiglu"

    # Research interfaces retained but disabled until their own experiments.
    latent_slots: int = 0
    mtp_heads: int = 0
    use_verifier_head: bool = False
    use_adaptive_halting: bool = False

    def __post_init__(self) -> None:
        for name in ("prelude_layers", "recurrent_layers", "recurrent_passes", "coda_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if len(self.core_mixer_schedule) != self.recurrent_layers:
            raise ValueError(
                "core_mixer_schedule length must equal recurrent_layers: "
                f"got {len(self.core_mixer_schedule)} and {self.recurrent_layers}"
            )
        unknown = set(self.core_mixer_schedule) - VALID_MIXERS
        if self.prelude_mixer not in VALID_MIXERS:
            unknown.add(self.prelude_mixer)
        if self.coda_mixer not in VALID_MIXERS:
            unknown.add(self.coda_mixer)
        if unknown:
            raise ValueError(f"unsupported mixer(s): {sorted(unknown)}")

        if self.recurrent_delta_scale <= 0:
            raise ValueError("recurrent_delta_scale must be positive")
        if not 0 < self.gdn_key_width_ratio <= 1:
            raise ValueError("gdn_key_width_ratio must be in (0, 1]")
        if self.gdn_head_dim <= 0 or self.gdn_expand_v <= 0:
            raise ValueError("Gated DeltaNet dimensions must be positive")
        if min(self.nsa_block_size, self.nsa_block_count, self.nsa_window_size) <= 0:
            raise ValueError("NSA block/window settings must be positive")
        if self.attention_position_strategy not in {"rope", "path"}:
            raise ValueError("attention_position_strategy must be 'rope' or 'path'")
        if self.ffn_kind not in {"dense_swiglu", "shared_routed_moe"}:
            raise ValueError("unsupported ffn_kind")
        if self.latent_slots < 0 or self.mtp_heads < 0:
            raise ValueError("latent_slots and mtp_heads cannot be negative")

    @classmethod
    def phi_control(cls) -> "IQArchitectureConfig":
        """Control topology: recurrent depth with Phi-compatible mixers only."""
        return cls(
            core_mixer_schedule=("phi_gqa",) * 8,
            attention_position_strategy="rope",
            ffn_kind="dense_swiglu",
        )

    @property
    def physical_layers(self) -> int:
        return self.prelude_layers + self.recurrent_layers + self.coda_layers

    @property
    def effective_depth(self) -> int:
        return (
            self.prelude_layers
            + self.recurrent_layers * self.recurrent_passes
            + self.coda_layers
        )

    def mixer_for_core_block(self, block_index: int) -> str:
        if not 0 <= block_index < self.recurrent_layers:
            raise IndexError(block_index)
        return self.core_mixer_schedule[block_index]

    def teacher_layer_for_prelude(self, block_index: int) -> int:
        if not 0 <= block_index < self.prelude_layers:
            raise IndexError(block_index)
        return block_index

    def teacher_layer_for_core(self, pass_index: int, block_index: int) -> int:
        if not 0 <= pass_index < self.recurrent_passes:
            raise IndexError(pass_index)
        if not 0 <= block_index < self.recurrent_layers:
            raise IndexError(block_index)
        return self.prelude_layers + pass_index * self.recurrent_layers + block_index

    def teacher_layer_for_coda(self, block_index: int) -> int:
        if not 0 <= block_index < self.coda_layers:
            raise IndexError(block_index)
        return self.prelude_layers + self.recurrent_layers * self.recurrent_passes + block_index

    def validate_teacher_depth(self, teacher_layers: int) -> None:
        if teacher_layers != self.effective_depth:
            raise ValueError(
                f"teacher depth {teacher_layers} does not match IQ effective depth {self.effective_depth}"
            )
