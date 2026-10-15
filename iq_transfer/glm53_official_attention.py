"""Load and evaluate an *official* GLM-5.3 MLA/DSA attention stage.

Uses Hugging Face Transformers' GlmMoeDsaAttention verbatim. No independent
imitation of MLA, indexer kernels or RoPE, and no IQ CSA weight-copy claim.
Requires transformers==5.15.0 for the pinned publisher checkpoint family.

https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import SafetensorsSource
from .glm53_attention_contract import GLM53AttentionGeometry
from .glm53_parallel_contract import GLM53Topology


class GLM53OfficialAttentionError(RuntimeError):
    pass


def _hf_attention_classes():
    try:
        from importlib.metadata import version
        if version("transformers") != "5.15.0":
            raise GLM53OfficialAttentionError(
                "official donor attention harness is pinned to transformers==5.15.0"
            )
        from transformers import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaAttention,
            GlmMoeDsaRotaryEmbedding,
        )
        return GlmMoeDsaConfig, GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding
    except ImportError as exc:
        raise GLM53OfficialAttentionError(
            "install transformers==5.15.0 for actual GLM attention semantics"
        ) from exc


def load_official_glm53_attention_layer(
    donor_dir: str | Path,
    *,
    layer: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load only the selected source attention layer from *real* BF16 shards.

    This is a teacher module; its hidden width remains 6144 and it cannot
    be dropped into IQ's 4096-wide CSA branch without validated adapters.
    """
    root = Path(donor_dir)
    config_path = root / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    topology = GLM53Topology.from_hf_config(raw)
    geometry = GLM53AttentionGeometry.from_hf_config(raw)
    if type(layer) is not int or not 0 <= layer < 78:
        raise GLM53OfficialAttentionError("teacher layer must be within 0..77")
    Config, Attention, _ = _hf_attention_classes()
    # BF16 files have already been decoded. Drop the FP8 loader hints so
    # Transformers does not attempt a second quantization pass.
    sanitized = dict(raw)
    sanitized.pop("quantization_config", None)
    config = Config(**sanitized)
    config._attn_implementation = "eager"
    # Instantiate on meta to avoid allocating two giant copies while reading
    # source weights stagewise.
    with torch.device("meta"):
        module = Attention(config, layer_idx=layer)
    module = module.to_empty(device=device)
    module = module.to(dtype=dtype)
    source = SafetensorsSource(root)
    required = geometry.source_shapes_for_block(
        layer, indexer_full=topology.indexer_types[layer] == "full"
    )
    expected_parameter_names = {key.split(".self_attn.", 1)[1] for key in required}
    actual_names = set(module.state_dict())
    if expected_parameter_names != actual_names:
        raise GLM53OfficialAttentionError(
            "upstream attention state keys differ from GLM5Bridge donor "
            f"missing={sorted(expected_parameter_names - actual_names)[:6]} "
            f"unexpected={sorted(actual_names - expected_parameter_names)[:6]}"
        )
    with torch.no_grad():
        targets = dict(module.named_parameters())
        targets.update(dict(module.named_buffers()))
        for source_key, shape in sorted(required.items()):
            local_name = source_key.split(".self_attn.", 1)[1]
            if source.shape(source_key) != shape:
                raise GLM53OfficialAttentionError(
                    f"unexpected donor MLA/DSA tensor geometry: {source_key}"
                )
            tensor = source.get(source_key)
            if tensor.dtype not in (torch.bfloat16, torch.float32):
                raise GLM53OfficialAttentionError(
                    f"{source_key} is not BF16/FP32; dequantize official FP8 "
                    "via GLM5Bridge before attempting teacher-layer load"
                )
            target = targets[local_name]
            if tuple(tensor.shape) != tuple(target.shape):
                raise GLM53OfficialAttentionError(
                    f"upstream attention shape mismatch: {local_name}"
                )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    module.eval()
    return module


def official_attention_prefill(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    previous_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Official eager MLA/DSA teacher prefill for one unpadded causal sequence.

    No KV-cache, no packed CP; these have distinct source validation needs.
    Returns the official attention output and actual DSA top-k indices.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != layer.config.hidden_size:
        raise GLM53OfficialAttentionError("teacher hidden states shape is invalid")
    batch, sequence, _ = hidden_states.shape
    if sequence < 1:
        raise GLM53OfficialAttentionError("teacher requires nonempty sequence")
    if position_ids is None:
        position_ids = torch.arange(
            sequence, device=hidden_states.device
        ).unsqueeze(0).expand(batch, -1)
    if position_ids.shape != (batch, sequence):
        raise GLM53OfficialAttentionError("position_ids must be [batch, sequence]")
    _, _, Rotary = _hf_attention_classes()
    rotary = Rotary(layer.config).to(device=hidden_states.device)
    cos_sin = rotary(hidden_states, position_ids)
    allowed = torch.ones(
        sequence, sequence, device=hidden_states.device, dtype=torch.bool
    ).tril()
    # HF eager attention combines additive causal + top-k sparse masks.
    attention_mask = torch.zeros(
        batch, 1, sequence, sequence,
        device=hidden_states.device, dtype=hidden_states.dtype,
    ).masked_fill(~allowed.view(1, 1, sequence, sequence),
                  torch.finfo(hidden_states.dtype).min)
    output, _, indices = layer(
        hidden_states, cos_sin, attention_mask,
        position_ids=position_ids, prev_topk_indices=previous_indices,
    )
    return output, indices
