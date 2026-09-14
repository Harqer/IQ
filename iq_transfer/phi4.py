from __future__ import annotations

from .donor import DonorConfig, DonorError, OperatorRef, TensorSource


class Phi4Inspector:
    """Inspector for Phi-4 / Phi-4-mini checkpoints using the Phi3 fused tensor layout."""

    SUPPORTED_MODEL_TYPES = {"phi3"}

    def __init__(self, config: DonorConfig) -> None:
        if config.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise DonorError(
                f"unsupported Phi-4 layout model_type={config.model_type!r}; "
                "this inspector currently supports Phi3ForCausalLM-style Phi-4 checkpoints"
            )
        self.config = config
        _ = config.head_dim

    @classmethod
    def from_config_mapping(cls, data):
        return cls(DonorConfig.from_mapping(data))

    def operators(self, source: TensorSource) -> tuple[OperatorRef, ...]:
        refs: list[OperatorRef] = []
        q_rows = self.config.num_attention_heads * self.config.head_dim
        kv_rows = self.config.num_key_value_heads * self.config.head_dim
        expected_qkv = q_rows + 2 * kv_rows
        expected_gate_up = 2 * self.config.intermediate_size

        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            qkv_key = f"{prefix}.self_attn.qkv_proj.weight"
            o_key = f"{prefix}.self_attn.o_proj.weight"
            gate_up_key = f"{prefix}.mlp.gate_up_proj.weight"
            down_key = f"{prefix}.mlp.down_proj.weight"

            self._expect_matrix(source, qkv_key, expected_qkv, self.config.hidden_size)
            self._expect_matrix(source, o_key, self.config.hidden_size, q_rows)
            self._expect_matrix(source, gate_up_key, expected_gate_up, self.config.hidden_size)
            self._expect_matrix(source, down_key, self.config.hidden_size, self.config.intermediate_size)

            refs.extend(
                [
                    OperatorRef(layer, "attn.q", qkv_key, (q_rows, self.config.hidden_size), 0, q_rows),
                    OperatorRef(layer, "attn.k", qkv_key, (kv_rows, self.config.hidden_size), q_rows, q_rows + kv_rows),
                    OperatorRef(layer, "attn.v", qkv_key, (kv_rows, self.config.hidden_size), q_rows + kv_rows, expected_qkv),
                    OperatorRef(layer, "attn.o", o_key, (self.config.hidden_size, q_rows)),
                    OperatorRef(layer, "mlp.gate", gate_up_key, (self.config.intermediate_size, self.config.hidden_size), 0, self.config.intermediate_size),
                    OperatorRef(layer, "mlp.up", gate_up_key, (self.config.intermediate_size, self.config.hidden_size), self.config.intermediate_size, expected_gate_up),
                    OperatorRef(layer, "mlp.down", down_key, (self.config.hidden_size, self.config.intermediate_size)),
                ]
            )
        return tuple(refs)

    @staticmethod
    def _expect_matrix(source: TensorSource, key: str, rows: int, cols: int) -> None:
        keys = source.keys()
        if key not in keys:
            raise DonorError(f"missing Phi-4 checkpoint tensor: {key}")
        shape = source.shape(key)
        if shape != (rows, cols):
            raise DonorError(f"unexpected shape for {key}: got {shape}, expected {(rows, cols)}")
