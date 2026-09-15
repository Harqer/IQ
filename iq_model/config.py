from __future__ import annotations

from dataclasses import dataclass


VALID_MIXERS = {
    "phi_gqa",
    "mamba3_mimo",
    "gated_deltanet",
    "nsa",
    "path_attention",
    "latent_nsa",
}


@dataclass(frozen=True)
class IQArchitectureConfig:
    """Topology for IQ Mamba hybrid v2.

    Mamba-3 MIMO is the primary sequence-state operator. Sparse attention anchors
    periodically recover precise long-range context that a fixed-size recurrent
    state can lose. Depth recurrence is a separate reasoning axis: the same physical
    middle stack is revisited across passes.

    The default intentionally does not combine every research idea at once. PaTH,
    latent NSA/MLA, Gated DeltaNet, MoE, and adaptive depth remain controlled
    ablations around the Mamba-3 + NSA + depth-recurrence backbone.
    """

    prelude_layers: int = 4
    recurrent_layers: int = 8
    recurrent_passes: int = 3
    coda_layers: int = 4

    # Primary v2 schedule: five Mamba-3 recurrent operators and three exact/sparse
    # retrieval anchors. This preserves the v1 5:3 operator budget for clean A/Bs.
    core_mixer_schedule: tuple[str, ...] = (
        "mamba3_mimo",
        "mamba3_mimo",
        "nsa",
        "mamba3_mimo",
        "mamba3_mimo",
        "nsa",
        "mamba3_mimo",
        "nsa",
    )

    # Unique boundary blocks stay Phi-compatible while cross-architecture transfer
    # is established. They are transfer scaffolding, not a claim that the final IQ
    # architecture must retain Phi attention at its boundaries.
    prelude_mixer: str = "phi_gqa"
    coda_mixer: str = "phi_gqa"

    # Depth recurrence.
    use_pass_embeddings: bool = True
    recurrent_delta_scale: float = 1.0

    # Mamba-3 MIMO. Defaults follow the official reference configuration family.
    mamba3_state_size: int = 128
    mamba3_head_dim: int = 64
    mamba3_expand: int = 2
    mamba3_mimo_rank: int = 4
    mamba3_chunk_size: int = 16
    mamba3_rope_fraction: float = 0.5
    mamba3_outproj_norm: bool = False

    # Gated DeltaNet remains an A/B control inherited from hybrid-v1.
    gdn_key_width_ratio: float = 0.75
    gdn_head_dim: int = 128
    gdn_expand_v: float = 2.0
    gdn_use_short_conv: bool = True
    gdn_conv_size: int = 4

    # Native Sparse Attention anchors.
    nsa_block_size: int = 64
    nsa_block_count: int = 16
    nsa_window_size: int = 512

    # Position strategy only applies to attention-family experiments. Mamba-3 owns
    # its state-space positional dynamics; PaTH is not injected into Mamba states.
    attention_position_strategy: str = "rope"

    # Dense SwiGLU stays the initial FFN control. Shared+routed MoE is a later
    # independent experiment after the Mamba-3 transfer is measurable.
    ffn_kind: str = "dense_swiglu"

    # JEPA-style latent prediction is an auxiliary representation objective. The
    # predictor can be enabled without changing the sequence mixer.
    use_latent_predictor: bool = True
    latent_predictor_dim: int = 0  # 0 => hidden_size

    # Research interfaces retained but disabled until their own controlled tests.
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

        if min(
            self.mamba3_state_size,
            self.mamba3_head_dim,
            self.mamba3_expand,
            self.mamba3_mimo_rank,
            self.mamba3_chunk_size,
        ) <= 0:
            raise ValueError("Mamba-3 dimensions must be positive")
        if not 0.0 <= self.mamba3_rope_fraction <= 1.0:
            raise ValueError("mamba3_rope_fraction must be in [0, 1]")
        if self.mamba3_chunk_size * self.mamba3_mimo_rank != 64:
            raise ValueError(
                "bf16 Mamba-3 MIMO reference kernels expect chunk_size * mimo_rank == 64"
            )

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
        if self.latent_predictor_dim < 0:
            raise ValueError("latent_predictor_dim cannot be negative")
        if self.latent_slots < 0 or self.mtp_heads < 0:
            raise ValueError("latent_slots and mtp_heads cannot be negative")

    @classmethod
    def phi_control(cls) -> "IQArchitectureConfig":
        """Recurrent-depth control with Phi-compatible sequence mixers only."""
        return cls(
            core_mixer_schedule=("phi_gqa",) * 8,
            attention_position_strategy="rope",
            ffn_kind="dense_swiglu",
            use_latent_predictor=False,
        )

    @classmethod
    def gated_deltanet_control(cls) -> "IQArchitectureConfig":
        """Hybrid-v1 control using the same retrieval-anchor schedule."""
        return cls(
            core_mixer_schedule=(
                "gated_deltanet",
                "gated_deltanet",
                "nsa",
                "gated_deltanet",
                "gated_deltanet",
                "nsa",
                "gated_deltanet",
                "nsa",
            ),
            use_latent_predictor=False,
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
