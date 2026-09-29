from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .donor import (
    DonorConfig,
    DonorError,
    LayerRef,
    OperatorRef,
    TensorSource,
    ValidationReport,
)


@dataclass(frozen=True)
class GLM53Layout:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    scoring_func: str
    index_head_dim: int
    index_n_heads: int
    indexer_types: tuple[str, ...]

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, num_layers: int) -> "GLM53Layout":
        required = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "first_k_dense_replace",
            "index_head_dim",
            "index_n_heads",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise DonorError(
                "missing GLM-5.3 config fields: " + ", ".join(missing)
            )
        indexer_types_raw = data.get("indexer_types")
        if indexer_types_raw is None:
            indexer_types = tuple("full" for _ in range(num_layers))
        else:
            if not isinstance(indexer_types_raw, list) or len(indexer_types_raw) < num_layers:
                raise DonorError(
                    "GLM-5.3 indexer_types must cover every backbone layer"
                )
            indexer_types = tuple(str(x) for x in indexer_types_raw[:num_layers])
        layout = cls(
            q_lora_rank=int(data["q_lora_rank"]),
            kv_lora_rank=int(data["kv_lora_rank"]),
            qk_nope_head_dim=int(data["qk_nope_head_dim"]),
            qk_rope_head_dim=int(data["qk_rope_head_dim"]),
            v_head_dim=int(data["v_head_dim"]),
            moe_intermediate_size=int(data["moe_intermediate_size"]),
            n_routed_experts=int(data["n_routed_experts"]),
            n_shared_experts=int(data["n_shared_experts"]),
            num_experts_per_tok=int(data["num_experts_per_tok"]),
            first_k_dense_replace=int(data["first_k_dense_replace"]),
            scoring_func=str(data.get("scoring_func", "")),
            index_head_dim=int(data["index_head_dim"]),
            index_n_heads=int(data["index_n_heads"]),
            indexer_types=indexer_types,
        )
        positive = {
            "q_lora_rank": layout.q_lora_rank,
            "kv_lora_rank": layout.kv_lora_rank,
            "qk_nope_head_dim": layout.qk_nope_head_dim,
            "v_head_dim": layout.v_head_dim,
            "moe_intermediate_size": layout.moe_intermediate_size,
            "n_routed_experts": layout.n_routed_experts,
            "n_shared_experts": layout.n_shared_experts,
            "num_experts_per_tok": layout.num_experts_per_tok,
            "index_head_dim": layout.index_head_dim,
            "index_n_heads": layout.index_n_heads,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise DonorError(
                "positive GLM-5.3 dimensions required: " + ", ".join(bad)
            )
        if layout.qk_rope_head_dim < 0:
            raise DonorError("qk_rope_head_dim cannot be negative")
        if not 0 <= layout.first_k_dense_replace <= num_layers:
            raise DonorError("first_k_dense_replace is outside the layer range")
        if layout.num_experts_per_tok > layout.n_routed_experts:
            raise DonorError("num_experts_per_tok exceeds n_routed_experts")
        if layout.n_shared_experts != 1:
            raise DonorError(
                "GLM-5.3 checkpoint layout currently requires one shared expert"
            )
        if layout.scoring_func != "sigmoid":
            raise DonorError(
                "GLM-5.3 inspector requires sigmoid MoE routing"
            )
        return layout


class GLM53Inspector:
    """Strict inspector for the GLM-5.3 glm_moe_dsa donor layout.

    The inspector exposes source operators without pretending GLM MLA/DSA or
    MoE tensors are shape-compatible with IQ. Transfer policy decides whether
    each source is operator-transported, functionally distilled, or ignored.
    """

    SUPPORTED_MODEL_TYPES = {"glm_moe_dsa"}

    def __init__(
        self,
        config: DonorConfig,
        layout: GLM53Layout,
    ) -> None:
        if config.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise DonorError(
                f"unsupported GLM-5.3 model_type={config.model_type!r}"
            )
        self.config = config
        self.layout = layout

    @classmethod
    def from_config_mapping(cls, data: Mapping[str, Any]) -> "GLM53Inspector":
        config = DonorConfig.from_mapping(data)
        return cls(
            config,
            GLM53Layout.from_mapping(
                data,
                num_layers=config.num_hidden_layers,
            ),
        )

    def layers(self) -> tuple[LayerRef, ...]:
        return tuple(
            LayerRef(i, f"model.layers.{i}")
            for i in range(self.config.num_hidden_layers)
        )

    def validate_checkpoint(self, source: TensorSource) -> ValidationReport:
        try:
            self.operators(source)
        except DonorError as exc:
            return ValidationReport(errors=(str(exc),))
        warnings: list[str] = []
        quantized = any(
            key.endswith(".weight_scale")
            or key.endswith(".weight_scale_inv")
            for key in source.keys()
        )
        if quantized:
            warnings.append(
                "quantized GLM checkpoint detected; use GLM-5.3-BF16 "
                "for canonical IQ weight transport"
            )
        return ValidationReport(warnings=tuple(warnings))

    def operators(self, source: TensorSource) -> tuple[OperatorRef, ...]:
        refs: list[OperatorRef] = []
        hidden = self.config.hidden_size
        heads = self.config.num_attention_heads
        q_width = heads * self.layout.qk_head_dim
        kv_width = heads * (
            self.layout.qk_nope_head_dim + self.layout.v_head_dim
        )

        self._add(
            refs, source, -1, "embedding",
            "model.embed_tokens.weight",
            self.config.vocab_size, hidden,
        )
        self._add_vector(
            refs, source, -1, "norm.final", "model.norm.weight", hidden
        )
        self._add(
            refs, source, -1, "lm_head",
            "lm_head.weight",
            self.config.vocab_size, hidden,
        )

        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            self._add_vector(
                refs, source, layer, "norm.input",
                f"{prefix}.input_layernorm.weight", hidden,
            )
            self._add_vector(
                refs, source, layer, "norm.post_attention",
                f"{prefix}.post_attention_layernorm.weight", hidden,
            )
            self._add(
                refs, source, layer, "attn.q_a",
                f"{prefix}.self_attn.q_a_proj.weight",
                self.layout.q_lora_rank, hidden,
            )
            self._add_vector(
                refs, source, layer, "attn.q_a_norm",
                f"{prefix}.self_attn.q_a_layernorm.weight",
                self.layout.q_lora_rank,
            )
            self._add(
                refs, source, layer, "attn.q_b",
                f"{prefix}.self_attn.q_b_proj.weight",
                q_width, self.layout.q_lora_rank,
            )
            self._add(
                refs, source, layer, "attn.kv_a_mqa",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                self.layout.kv_lora_rank + self.layout.qk_rope_head_dim,
                hidden,
            )
            self._add_vector(
                refs, source, layer, "attn.kv_a_norm",
                f"{prefix}.self_attn.kv_a_layernorm.weight",
                self.layout.kv_lora_rank,
            )
            self._add(
                refs, source, layer, "attn.kv_b",
                f"{prefix}.self_attn.kv_b_proj.weight",
                kv_width, self.layout.kv_lora_rank,
            )
            self._add(
                refs, source, layer, "attn.o",
                f"{prefix}.self_attn.o_proj.weight",
                hidden, heads * self.layout.v_head_dim,
            )

            if self.layout.indexer_types[layer] != "shared":
                indexer = f"{prefix}.self_attn.indexer"
                self._add(
                    refs, source, layer, "dsa.indexer.q",
                    f"{indexer}.wq_b.weight",
                    self.layout.index_n_heads * self.layout.index_head_dim,
                    self.layout.q_lora_rank,
                )
                self._add(
                    refs, source, layer, "dsa.indexer.k",
                    f"{indexer}.wk.weight",
                    self.layout.index_head_dim,
                    hidden,
                )
                self._add(
                    refs, source, layer, "dsa.indexer.head_weights",
                    f"{indexer}.weights_proj.weight",
                    self.layout.index_n_heads,
                    hidden,
                )
                self._add_vector(
                    refs, source, layer, "dsa.indexer.k_norm",
                    f"{indexer}.k_norm.weight",
                    self.layout.index_head_dim,
                )
                self._add_vector(
                    refs, source, layer, "dsa.indexer.k_norm_bias",
                    f"{indexer}.k_norm.bias",
                    self.layout.index_head_dim,
                )

            if layer < self.layout.first_k_dense_replace:
                self._add(
                    refs, source, layer, "mlp.gate",
                    f"{prefix}.mlp.gate_proj.weight",
                    self.config.intermediate_size, hidden,
                )
                self._add(
                    refs, source, layer, "mlp.up",
                    f"{prefix}.mlp.up_proj.weight",
                    self.config.intermediate_size, hidden,
                )
                self._add(
                    refs, source, layer, "mlp.down",
                    f"{prefix}.mlp.down_proj.weight",
                    hidden, self.config.intermediate_size,
                )
            else:
                self._add(
                    refs, source, layer, "moe.router",
                    f"{prefix}.mlp.gate.weight",
                    self.layout.n_routed_experts, hidden,
                )
                for shared in range(self.layout.n_shared_experts):
                    shared_prefix = f"{prefix}.mlp.shared_experts"
                    role_prefix = (
                        "moe.shared"
                        if self.layout.n_shared_experts == 1
                        else f"moe.shared.{shared}"
                    )
                    self._add(
                        refs, source, layer, f"{role_prefix}.gate",
                        f"{shared_prefix}.gate_proj.weight",
                        self.layout.moe_intermediate_size, hidden,
                    )
                    self._add(
                        refs, source, layer, f"{role_prefix}.up",
                        f"{shared_prefix}.up_proj.weight",
                        self.layout.moe_intermediate_size, hidden,
                    )
                    self._add(
                        refs, source, layer, f"{role_prefix}.down",
                        f"{shared_prefix}.down_proj.weight",
                        hidden, self.layout.moe_intermediate_size,
                    )
                for expert in range(self.layout.n_routed_experts):
                    expert_prefix = f"{prefix}.mlp.experts.{expert}"
                    role_prefix = f"moe.expert.{expert}"
                    self._add(
                        refs, source, layer, f"{role_prefix}.gate",
                        f"{expert_prefix}.gate_proj.weight",
                        self.layout.moe_intermediate_size, hidden,
                    )
                    self._add(
                        refs, source, layer, f"{role_prefix}.up",
                        f"{expert_prefix}.up_proj.weight",
                        self.layout.moe_intermediate_size, hidden,
                    )
                    self._add(
                        refs, source, layer, f"{role_prefix}.down",
                        f"{expert_prefix}.down_proj.weight",
                        hidden, self.layout.moe_intermediate_size,
                    )

        return tuple(refs)

    @staticmethod
    def _expect(
        source: TensorSource,
        key: str,
        shape: tuple[int, ...],
    ) -> None:
        if key not in source.keys():
            raise DonorError(f"missing GLM-5.3 checkpoint tensor: {key}")
        actual = source.shape(key)
        if actual != shape:
            raise DonorError(
                f"unexpected shape for {key}: got {actual}, expected {shape}"
            )

    @classmethod
    def _add(
        cls,
        refs: list[OperatorRef],
        source: TensorSource,
        layer: int,
        role: str,
        key: str,
        rows: int | None,
        cols: int,
    ) -> None:
        if rows is None:
            raise DonorError(
                f"GLM-5.3 {role} requires vocab_size in donor config"
            )
        shape = (rows, cols)
        cls._expect(source, key, shape)
        refs.append(OperatorRef(layer, role, key, shape))

    @classmethod
    def _add_vector(
        cls,
        refs: list[OperatorRef],
        source: TensorSource,
        layer: int,
        role: str,
        key: str,
        size: int,
    ) -> None:
        shape = (size,)
        cls._expect(source, key, shape)
        refs.append(OperatorRef(layer, role, key, shape))
