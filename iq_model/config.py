from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


class IQConfigError(ValueError):
    pass


@dataclass(frozen=True)
class IQModelConfig:
    vocab_size: int
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    max_position_embeddings: int = 32768
    rope_theta: float = 250000.0
    rotary_fraction: float = 1.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # Mamba-3 is an IQ production architecture component. MIMO is mandatory;
    # there is intentionally no is_mimo feature flag or SISO fallback.
    mamba3_state_size: int = 128
    mamba3_head_dim: int = 64
    mamba3_mimo_rank: int = 4
    mamba3_expand: float = 2.0
    mamba3_rope_fraction: float = 0.5
    mamba3_chunk_size: int = 16
    mamba3_outproj_norm: bool = False

    def __post_init__(self) -> None:
        ints = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "mamba3_state_size": self.mamba3_state_size,
            "mamba3_head_dim": self.mamba3_head_dim,
            "mamba3_mimo_rank": self.mamba3_mimo_rank,
            "mamba3_chunk_size": self.mamba3_chunk_size,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise IQConfigError(f"positive integer fields required: {', '.join(bad)}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise IQConfigError("hidden_size must be divisible by num_attention_heads")
        mamba_inner = int(self.mamba3_expand * self.hidden_size)
        if self.mamba3_expand <= 0:
            raise IQConfigError("mamba3_expand must be positive")
        if mamba_inner % self.mamba3_head_dim != 0:
            raise IQConfigError(
                "Mamba-3 inner width must be divisible by mamba3_head_dim"
            )
        if self.mamba3_mimo_rank < 2:
            raise IQConfigError(
                "IQ requires Mamba-3 MIMO; mamba3_mimo_rank must be >= 2"
            )
        if self.mamba3_rope_fraction not in (0.5, 1.0):
            raise IQConfigError("mamba3_rope_fraction must be 0.5 or 1.0")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise IQConfigError("num_attention_heads must be divisible by num_key_value_heads")
        if not (0.0 < self.rotary_fraction <= 1.0):
            raise IQConfigError("rotary_fraction must be in (0, 1]")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim or self.rotary_dim % 2 != 0:
            raise IQConfigError("rotary dimension must be positive, even, and <= head_dim")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0 or self.initializer_range <= 0:
            raise IQConfigError("rope_theta, rms_norm_eps, and initializer_range must be positive")
        for name in ("attention_dropout", "residual_dropout"):
            value = float(getattr(self, name))
            if not (0.0 <= value < 1.0):
                raise IQConfigError(f"{name} must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        dim = int(self.head_dim * self.rotary_fraction)
        return dim - (dim % 2)

    @property
    def kv_repeat(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "IQModelConfig":
        try:
            return cls(**data)
        except TypeError as exc:
            raise IQConfigError(f"invalid IQ model config: {exc}") from exc

    @classmethod
    def from_json(cls, path: str) -> "IQModelConfig":
        from pathlib import Path
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IQConfigError(f"invalid IQ model config file: {path}") from exc
        if not isinstance(data, dict):
            raise IQConfigError("IQ model config JSON must contain an object")
        return cls.from_dict(data)

    def write_json(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(payload).hexdigest()
