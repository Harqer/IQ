from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IQArchitectureConfig:
    """Topology for the first research-valid IQ backbone.

    v0 deliberately keeps Phi-compatible token, attention, FFN, norm, and RoPE
    mechanics. The architectural experiment is recurrent depth: unique prelude and
    coda blocks surrounding a shared recurrent middle core.
    """

    prelude_layers: int = 4
    recurrent_layers: int = 8
    recurrent_passes: int = 3
    coda_layers: int = 4

    # Zero-initialized pass embeddings give the recurrent core a way to distinguish
    # visits without perturbing a transplanted donor at initialization time.
    use_pass_embeddings: bool = True

    # Keep 1.0 for the first transfer experiment so the Phi block function is not
    # altered implicitly. Recurrence-aware scaling is an explicit ablation rather
    # than a hidden behavior change.
    recurrent_delta_scale: float = 1.0

    # Research interfaces. These are intentionally disabled for v0 transfer.
    latent_slots: int = 0
    mtp_heads: int = 0
    use_verifier_head: bool = False
    use_adaptive_halting: bool = False

    def __post_init__(self) -> None:
        for name in ("prelude_layers", "recurrent_layers", "recurrent_passes", "coda_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.recurrent_delta_scale <= 0:
            raise ValueError("recurrent_delta_scale must be positive")
        if self.latent_slots < 0 or self.mtp_heads < 0:
            raise ValueError("latent_slots and mtp_heads cannot be negative")

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
