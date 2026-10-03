from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import gc
import json

from .batches import TokenBatchArtifact
from .capture_runner import ActivationBundle
from .checkpoint import SafetensorsSource
from .complete_transplant import canonical_glm53_config
from .donor import DonorError
from .glm53 import GLM53Inspector
from .glm53_calibration import bootstrap_glm53_source_layer_map


class GLM53StreamingCaptureError(RuntimeError):
    pass


class HubSafetensorsSource(SafetensorsSource):
    """Safetensors source that downloads missing indexed shards on demand."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        repo_id: str,
        revision: str,
        token: str | None = None,
    ) -> None:
        super().__init__(model_dir)
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self._ephemeral_shards: set[Path] = set()
        self._shard_last_layer: dict[str, int] = {}
        for key, shard in self._weight_map.items():
            parts = key.split(".")
            last_layer = -1
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                try:
                    last_layer = int(parts[2])
                except ValueError:
                    last_layer = -1
            self._shard_last_layer[shard] = max(
                last_layer,
                self._shard_last_layer.get(shard, -1),
            )

    def _path(self, key: str) -> Path:
        try:
            shard = self._weight_map[key]
        except KeyError as exc:
            raise DonorError(f"unknown checkpoint tensor: {key}") from exc
        path = self.model_dir / shard
        if path.is_file():
            return path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise GLM53StreamingCaptureError(
                "huggingface_hub is required for streaming capture"
            ) from exc
        resolved = Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=shard,
                revision=self.revision,
                local_dir=self.model_dir,
                token=self.token,
            )
        )
        if resolved.resolve() != path.resolve() and not path.is_file():
            raise GLM53StreamingCaptureError(
                f"Hub download did not materialize expected shard: {shard}"
            )
        self._ephemeral_shards.add(path)
        return path

    def clear_ephemeral_shards(self, *, completed_layer: int | None = None) -> None:
        """Evict downloaded shards once no later layer can reference them."""
        for path in tuple(self._ephemeral_shards):
            last_layer = self._shard_last_layer.get(path.name, -1)
            if completed_layer is not None and last_layer > completed_layer:
                continue
            try:
                path.unlink(missing_ok=True)
            finally:
                self._ephemeral_shards.discard(path)
def _require_runtime():
    try:
        import torch
        import torch.nn.functional as F
        from transformers.masking_utils import create_causal_mask
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
            GlmMoeDsaConfig,
        )
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaAttention,
            GlmMoeDsaRotaryEmbedding,
        )
    except ImportError as exc:
        raise GLM53StreamingCaptureError(
            "streaming GLM capture requires torch and transformers==5.17.0"
        ) from exc
    return (
        torch,
        F,
        create_causal_mask,
        GlmMoeDsaConfig,
        GlmMoeDsaAttention,
        GlmMoeDsaRotaryEmbedding,
    )


def _rms_norm(hidden: Any, weight: Any, eps: float) -> Any:
    input_dtype = hidden.dtype
    normalized = hidden.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * (variance + eps).rsqrt()
    return weight * normalized.to(input_dtype)
def _load_attention(
    source: SafetensorsSource,
    *,
    config: Any,
    layer: int,
    device: str,
    attention_cls: Any,
) -> Any:
    torch, *_ = _require_runtime()
    with torch.device("meta"):
        module = attention_cls(config, layer)
    prefix = f"model.layers.{layer}.self_attn."
    state: dict[str, Any] = {}
    for local_key in module.state_dict():
        donor_key = prefix + local_key
        state[local_key] = source.get(donor_key).to(device)
    incompatible = module.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise GLM53StreamingCaptureError(
            "attention state mismatch: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    module.eval()
    return module


def _load_vector(
    source: SafetensorsSource,
    key: str,
    *,
    device: str,
) -> Any:
    value = source.get(key)
    if value.ndim != 1:
        raise GLM53StreamingCaptureError(f"expected vector tensor: {key}")
    return value.to(device)
def _route_sparse_moe(
    hidden: Any,
    *,
    router_weight: Any,
    correction_bias: Any,
    num_group: int,
    topk_group: int,
    top_k: int,
    normalize: bool,
    scaling: float,
) -> tuple[Any, Any]:
    torch, F, *_ = _require_runtime()
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat.float(), router_weight.float())
    scores = logits.sigmoid()
    scores_for_choice = scores + correction_bias.float()
    experts = scores_for_choice.shape[-1]
    if experts % num_group != 0:
        raise GLM53StreamingCaptureError(
            "router expert count must divide evenly across groups"
        )
    group_width = experts // num_group
    group_scores = (
        scores_for_choice.view(-1, num_group, group_width)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(
        group_scores,
        k=topk_group,
        dim=-1,
        sorted=False,
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        -1,
        num_group,
        group_width,
    ).reshape(-1, experts)
    scores_for_choice = scores_for_choice.masked_fill(
        ~score_mask.bool(),
        float("-inf"),
    )
    topk_indices = torch.topk(
        scores_for_choice,
        k=top_k,
        dim=-1,
        sorted=False,
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    if normalize:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    topk_weights = topk_weights * scaling
    return topk_weights, topk_indices


def _dense_mlp(
    source: SafetensorsSource,
    *,
    prefix: str,
    hidden: Any,
    device: str,
) -> tuple[Any, Any]:
    _, F, *_ = _require_runtime()
    gate_w = source.get(f"{prefix}.gate_proj.weight").to(device)
    up_w = source.get(f"{prefix}.up_proj.weight").to(device)
    down_w = source.get(f"{prefix}.down_proj.weight").to(device)
    gate = F.linear(hidden, gate_w)
    up = F.linear(hidden, up_w)
    intermediate = F.silu(gate) * up
    output = F.linear(intermediate, down_w)
    return output, intermediate
def _sparse_moe(
    source: SafetensorsSource,
    *,
    prefix: str,
    hidden: Any,
    config_data: Mapping[str, Any],
    device: str,
) -> tuple[Any, Any]:
    torch, F, *_ = _require_runtime()
    router_weight = source.get(f"{prefix}.gate.weight").to(device)
    correction_bias = source.get(
        f"{prefix}.gate.e_score_correction_bias"
    ).to(device)
    topk_weights, topk_indices = _route_sparse_moe(
        hidden,
        router_weight=router_weight,
        correction_bias=correction_bias,
        num_group=int(config_data["n_group"]),
        topk_group=int(config_data["topk_group"]),
        top_k=int(config_data["num_experts_per_tok"]),
        normalize=bool(config_data.get("norm_topk_prob", True)),
        scaling=float(config_data["routed_scaling_factor"]),
    )
    flat = hidden.reshape(-1, hidden.shape[-1])
    routed = torch.zeros_like(flat)
    expert_ids = torch.unique(topk_indices).tolist()
    for expert_id in sorted(int(x) for x in expert_ids):
        matches = (topk_indices == expert_id).nonzero(as_tuple=False)
        token_idx = matches[:, 0]
        topk_pos = matches[:, 1]
        expert_prefix = f"{prefix}.experts.{expert_id}"
        gate_w = source.get(f"{expert_prefix}.gate_proj.weight").to(device)
        up_w = source.get(f"{expert_prefix}.up_proj.weight").to(device)
        down_w = source.get(f"{expert_prefix}.down_proj.weight").to(device)
        current = flat[token_idx]
        current = F.silu(F.linear(current, gate_w)) * F.linear(current, up_w)
        current = F.linear(current, down_w)
        current = current * topk_weights[token_idx, topk_pos, None]
        routed.index_add_(0, token_idx, current.to(routed.dtype))
        del gate_w, up_w, down_w, current

    shared_prefix = f"{prefix}.shared_experts"
    shared_gate = source.get(f"{shared_prefix}.gate_proj.weight").to(device)
    shared_up = source.get(f"{shared_prefix}.up_proj.weight").to(device)
    shared_down = source.get(f"{shared_prefix}.down_proj.weight").to(device)
    shared = F.silu(F.linear(hidden, shared_gate)) * F.linear(hidden, shared_up)
    shared = F.linear(shared, shared_down)
    output = routed.view_as(hidden) + shared
    return output, topk_indices


def _append_valid(
    captures: dict[str, list[Any]],
    key: str,
    value: Any,
    valid_mask: Any,
) -> None:
    flat = value.reshape(-1, *value.shape[2:])
    selected = flat[valid_mask.reshape(-1)]
    captures.setdefault(key, []).append(
        selected.detach().contiguous().cpu()
    )


def _prepare_batch(
    batch: Mapping[str, Any],
    *,
    device: str,
) -> tuple[Any, Any, Any]:
    torch, *_ = _require_runtime()
    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.to(device=device).bool()
    position_ids = batch.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(
            input_ids.shape[1],
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
    else:
        position_ids = position_ids.to(device=device, dtype=torch.long)
    return input_ids, attention_mask, position_ids


def _causal_mask(
    *,
    config: Any,
    hidden: Any,
    attention_mask: Any,
    position_ids: Any,
    create_causal_mask: Any,
) -> Any:
    return create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
        allow_is_causal_skip=False,
    )


def capture_glm53_bootstrap_streaming(
    *,
    source: HubSafetensorsSource,
    config_path: str | Path,
    batches: TokenBatchArtifact | Iterable[Mapping[str, Any]],
    device: str = "cpu",
    attention_implementation: str = "eager",
) -> ActivationBundle:
    (
        torch,
        _,
        create_causal_mask,
        config_cls,
        attention_cls,
        rotary_cls,
    ) = _require_runtime()
    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise GLM53StreamingCaptureError("GLM config must contain a JSON object")
    inspector = GLM53Inspector.from_config_mapping(config_data)
    target = canonical_glm53_config()
    mapping = bootstrap_glm53_source_layer_map(
        source_layers=inspector.config.num_hidden_layers,
        target_config=target,
        source_indexer_types=inspector.layout.indexer_types,
    )
    capture_layers = frozenset(mapping.values())
    config = config_cls(**config_data)
    config._attn_implementation = attention_implementation
    rotary = rotary_cls(config=config).to(device)
    rotary.eval()

    batch_tuple = (
        batches.batches
        if isinstance(batches, TokenBatchArtifact)
        else tuple(dict(batch) for batch in batches)
    )
    if not batch_tuple:
        raise GLM53StreamingCaptureError(
            "at least one token batch is required"
        )

    captures: dict[str, list[Any]] = {}
    total_samples = 0
    with torch.no_grad():
        for batch in batch_tuple:
            if "document_ids" in batch:
                raise GLM53StreamingCaptureError(
                    "GLM bootstrap capture does not support packed document_ids"
                )
            input_ids, valid_mask, position_ids = _prepare_batch(
                batch,
                device=device,
            )
            embedding = source.get_rows(
                "model.embed_tokens.weight",
                input_ids.detach().cpu(),
            ).reshape(*input_ids.shape, inspector.config.hidden_size)
            hidden = embedding.to(device)
            total_samples += int(valid_mask.sum().item())
            causal = _causal_mask(
                config=config,
                hidden=hidden,
                attention_mask=valid_mask,
                position_ids=position_ids,
                create_causal_mask=create_causal_mask,
            )
            position_embeddings = rotary(
                hidden,
                position_ids=position_ids,
            )
            prev_attention_topk = None

            for layer in range(inspector.config.num_hidden_layers):
                prefix = f"model.layers.{layer}"
                attention = _load_attention(
                    source,
                    config=config,
                    layer=layer,
                    device=device,
                    attention_cls=attention_cls,
                )
                input_norm = _load_vector(
                    source,
                    f"{prefix}.input_layernorm.weight",
                    device=device,
                )
                residual = hidden
                attention_input = _rms_norm(
                    hidden,
                    input_norm,
                    float(config.rms_norm_eps),
                )
                compressed_kv = None
                if layer in capture_layers:
                    compressed_kv = attention.kv_a_proj_with_mqa(
                        attention_input
                    )
                attention_output, _, attention_topk = attention(
                    hidden_states=attention_input,
                    attention_mask=causal,
                    position_ids=position_ids,
                    past_key_values=None,
                    position_embeddings=position_embeddings,
                    prev_topk_indices=prev_attention_topk,
                )
                hidden = residual + attention_output
                prev_attention_topk = attention_topk

                post_norm = _load_vector(
                    source,
                    f"{prefix}.post_attention_layernorm.weight",
                    device=device,
                )
                residual = hidden
                mlp_input = _rms_norm(
                    hidden,
                    post_norm,
                    float(config.rms_norm_eps),
                )
                mlp_prefix = f"{prefix}.mlp"
                dense_hidden = None
                router_topk = None
                if layer < inspector.layout.first_k_dense_replace:
                    mlp_output, dense_hidden = _dense_mlp(
                        source,
                        prefix=mlp_prefix,
                        hidden=mlp_input,
                        device=device,
                    )
                else:
                    mlp_output, router_topk = _sparse_moe(
                        source,
                        prefix=mlp_prefix,
                        hidden=mlp_input,
                        config_data=config_data,
                        device=device,
                    )
                hidden = residual + mlp_output

                if layer in capture_layers:
                    assert compressed_kv is not None
                    _append_valid(
                        captures,
                        f"layer.{layer}.compressed_kv",
                        compressed_kv,
                        valid_mask,
                    )
                    if layer < inspector.layout.first_k_dense_replace:
                        assert dense_hidden is not None
                        _append_valid(
                            captures,
                            f"layer.{layer}.mlp_hidden",
                            dense_hidden,
                            valid_mask,
                        )
                    else:
                        assert router_topk is not None
                        router_view = router_topk.reshape(
                            *input_ids.shape,
                            -1,
                        )
                        _append_valid(
                            captures,
                            f"layer.{layer}.router_topk",
                            router_view,
                            valid_mask,
                        )

                del (
                    attention,
                    input_norm,
                    post_norm,
                    attention_input,
                    attention_output,
                    mlp_input,
                    mlp_output,
                    compressed_kv,
                    dense_hidden,
                    router_topk,
                )
                gc.collect()
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                source.clear_ephemeral_shards(completed_layer=layer)

            del hidden, embedding, causal, position_embeddings
            source.clear_ephemeral_shards()

    if not captures:
        raise GLM53StreamingCaptureError(
            "streaming capture produced no mapped calibration spaces"
        )
    spaces = {
        key: torch.cat(parts, dim=0).contiguous()
        for key, parts in captures.items()
    }
    for key, value in spaces.items():
        if int(value.shape[0]) != total_samples:
            raise GLM53StreamingCaptureError(
                f"capture sample mismatch for {key}: "
                f"{value.shape[0]} != {total_samples}"
            )
    return ActivationBundle(
        spaces=spaces,
        sample_count=total_samples,
        batch_count=len(batch_tuple),
    )
