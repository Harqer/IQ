from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import math

import torch


GPT_OSS_20B_REPO = "openai/gpt-oss-20b"
GPT_OSS_20B_REVISION = "0d1f28dce7d3a8794b20345f496dfabb28d51e70"
GPT_OSS_20B_ORIGINAL_SHA256 = "3340a61d1a0391e8c5b5d3463d18d4c48129a84bbc04a554c762c99020aa06ed"
FP4_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


class GptOss20BError(RuntimeError):
    pass


@dataclass(frozen=True)
class GptOss20BConfig:
    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    swiglu_limit: float
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int
    initial_context_length: int
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GptOss20BConfig":
        try:
            result = cls(
                num_hidden_layers=int(data["num_hidden_layers"]),
                num_experts=int(data["num_experts"]),
                experts_per_token=int(data["experts_per_token"]),
                vocab_size=int(data["vocab_size"]),
                hidden_size=int(data["hidden_size"]),
                intermediate_size=int(data["intermediate_size"]),
                swiglu_limit=float(data["swiglu_limit"]),
                head_dim=int(data["head_dim"]),
                num_attention_heads=int(data["num_attention_heads"]),
                num_key_value_heads=int(data["num_key_value_heads"]),
                sliding_window=int(data["sliding_window"]),
                initial_context_length=int(data["initial_context_length"]),
                rope_theta=float(data["rope_theta"]),
                rope_scaling_factor=float(data["rope_scaling_factor"]),
                rope_ntk_alpha=float(data["rope_ntk_alpha"]),
                rope_ntk_beta=float(data["rope_ntk_beta"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GptOss20BError(f"invalid gpt-oss original config: {exc}") from exc
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "GptOss20BConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GptOss20BError(f"cannot read gpt-oss config: {path}") from exc
        if not isinstance(data, dict):
            raise GptOss20BError("gpt-oss config must be a JSON object")
        return cls.from_mapping(data)

    def validate(self) -> None:
        expected = {
            "num_hidden_layers": 24,
            "num_experts": 32,
            "experts_per_token": 4,
            "vocab_size": 201088,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "swiglu_limit": 7.0,
            "head_dim": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "sliding_window": 128,
            "initial_context_length": 4096,
            "rope_theta": 150000.0,
            "rope_scaling_factor": 32.0,
            "rope_ntk_alpha": 1.0,
            "rope_ntk_beta": 32.0,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} expected={value!r}"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if mismatches:
            raise GptOss20BError(
                "checkpoint is not the documented gpt-oss-20b original architecture: "
                + "; ".join(mismatches)
            )

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class GptOss20BOriginalCheckpoint:
    """Lazy reader for OpenAI's original gpt-oss MXFP4 checkpoint.

    MoE matrices are dequantized one requested tensor at a time so the complete
    20B model is never resident in RAM during checkpoint compilation.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        verify_hash: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "config.json"
        self.weights_path = self.model_dir / "model.safetensors"
        if not self.config_path.exists() or not self.weights_path.exists():
            raise GptOss20BError(
                "gpt-oss original directory must contain config.json and model.safetensors"
            )
        self.config = GptOss20BConfig.from_json(self.config_path)
        self.sha256 = sha256_file(self.weights_path)
        if verify_hash and self.sha256 != GPT_OSS_20B_ORIGINAL_SHA256:
            raise GptOss20BError(
                "gpt-oss-20b original checkpoint SHA-256 mismatch: "
                f"got {self.sha256}, expected {GPT_OSS_20B_ORIGINAL_SHA256}"
            )
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise GptOss20BError("safetensors is required") from exc
        self._safe_open = safe_open
        with safe_open(self.weights_path, framework="pt", device="cpu") as handle:
            self._keys = frozenset(handle.keys())

    def keys(self) -> frozenset[str]:
        return self._keys

    def tensor(self, name: str) -> torch.Tensor:
        if name not in self._keys:
            raise GptOss20BError(f"missing gpt-oss tensor: {name}")
        with self._safe_open(
            self.weights_path,
            framework="pt",
            device="cpu",
        ) as handle:
            return handle.get_tensor(name)

    def _mxfp4(
        self,
        base_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 262144,
    ) -> torch.Tensor:
        blocks_name = base_name + ".blocks"
        scales_name = base_name + ".scales"
        if blocks_name not in self._keys or scales_name not in self._keys:
            raise GptOss20BError(
                f"missing MXFP4 blocks/scales for {base_name}"
            )
        blocks = self.tensor(blocks_name)
        scales = self.tensor(scales_name).to(torch.int32) - 127
        if blocks.shape[:-1] != scales.shape:
            raise GptOss20BError(
                f"MXFP4 shape mismatch for {base_name}: "
                f"blocks={tuple(blocks.shape)} scales={tuple(scales.shape)}"
            )

        lut = torch.tensor(FP4_VALUES, dtype=dtype)
        *prefix_shape, groups, packed = blocks.shape
        rows_total = math.prod(prefix_shape) * groups
        flat_blocks = blocks.reshape(rows_total, packed)
        flat_scales = scales.reshape(rows_total, 1)
        output = torch.empty(
            rows_total,
            packed * 2,
            dtype=dtype,
        )
        for start in range(0, rows_total, rows_per_chunk):
            stop = min(start + rows_per_chunk, rows_total)
            block = flat_blocks[start:stop]
            exponent = flat_scales[start:stop]
            low = (block & 0x0F).to(torch.long)
            high = (block >> 4).to(torch.long)
            target = output[start:stop]
            target[:, 0::2] = lut[low]
            target[:, 1::2] = lut[high]
            torch.ldexp(target, exponent, out=target)
        return output.reshape(
            *prefix_shape,
            groups,
            packed * 2,
        ).view(*prefix_shape, groups * packed * 2)

    def attention(self, layer: int) -> dict[str, torch.Tensor]:
        if not 0 <= layer < self.config.num_hidden_layers:
            raise GptOss20BError("attention layer outside donor depth")
        prefix = f"block.{layer}.attn"
        qkv_weight = self.tensor(f"{prefix}.qkv.weight")
        qkv_bias = self.tensor(f"{prefix}.qkv.bias")
        expected_weight = (self.config.qkv_dim, self.config.hidden_size)
        if tuple(qkv_weight.shape) != expected_weight:
            raise GptOss20BError(
                f"unexpected qkv weight shape {tuple(qkv_weight.shape)}"
            )
        q_end = self.config.q_dim
        k_end = q_end + self.config.kv_dim
        return {
            "norm": self.tensor(f"{prefix}.norm.scale"),
            "q_weight": qkv_weight[:q_end].contiguous(),
            "k_weight": qkv_weight[q_end:k_end].contiguous(),
            "v_weight": qkv_weight[k_end:].contiguous(),
            "q_bias": qkv_bias[:q_end].contiguous(),
            "k_bias": qkv_bias[q_end:k_end].contiguous(),
            "v_bias": qkv_bias[k_end:].contiguous(),
            "out_weight": self.tensor(f"{prefix}.out.weight"),
            "out_bias": self.tensor(f"{prefix}.out.bias"),
            "sinks": self.tensor(f"{prefix}.sinks"),
        }

    def moe(self, layer: int) -> dict[str, torch.Tensor]:
        if not 0 <= layer < self.config.num_hidden_layers:
            raise GptOss20BError("MoE layer outside donor depth")
        prefix = f"block.{layer}.mlp"
        mlp1 = self._mxfp4(f"{prefix}.mlp1_weight")
        mlp2 = self._mxfp4(f"{prefix}.mlp2_weight")
        expected_mlp1 = (
            self.config.num_experts,
            2 * self.config.intermediate_size,
            self.config.hidden_size,
        )
        expected_mlp2 = (
            self.config.num_experts,
            self.config.hidden_size,
            self.config.intermediate_size,
        )
        if tuple(mlp1.shape) != expected_mlp1:
            raise GptOss20BError(
                f"unexpected mlp1 shape {tuple(mlp1.shape)}"
            )
        if tuple(mlp2.shape) != expected_mlp2:
            raise GptOss20BError(
                f"unexpected mlp2 shape {tuple(mlp2.shape)}"
            )
        return {
            "norm": self.tensor(f"{prefix}.norm.scale"),
            "router_weight": self.tensor(f"{prefix}.gate.weight"),
            "router_bias": self.tensor(f"{prefix}.gate.bias"),
            "gate_weight": mlp1[:, 0::2, :].contiguous(),
            "up_weight": mlp1[:, 1::2, :].contiguous(),
            "gate_bias": self.tensor(f"{prefix}.mlp1_bias")[:, 0::2].contiguous(),
            "up_bias": self.tensor(f"{prefix}.mlp1_bias")[:, 1::2].contiguous(),
            "down_weight": mlp2.contiguous(),
            "down_bias": self.tensor(f"{prefix}.mlp2_bias"),
        }

    def globals(self) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.tensor("embedding.weight"),
            "norm": self.tensor("norm.scale"),
            "lm_head": self.tensor("unembedding.weight"),
        }
