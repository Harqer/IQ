from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .calibration import ActivationPair, PhiLayerCalibration
from .capture import ActivationTap, TorchActivationCapture


class CaptureRunnerError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise CaptureRunnerError("PyTorch is required for activation capture") from exc
    return torch


@dataclass(frozen=True)
class ActivationBundle:
    spaces: Mapping[str, Any]
    sample_count: int
    batch_count: int

    def __post_init__(self) -> None:
        torch = _require_torch()
        if self.sample_count <= 0 or self.batch_count <= 0:
            raise CaptureRunnerError("activation bundle counts must be positive")
        if not self.spaces:
            raise CaptureRunnerError("activation bundle must contain spaces")
        for name, value in self.spaces.items():
            if not str(name).strip():
                raise CaptureRunnerError("activation space names must be non-empty")
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise CaptureRunnerError(f"activation space {name!r} must be a rank-2 tensor")
            if value.shape[0] != self.sample_count:
                raise CaptureRunnerError(
                    f"activation space {name!r} has {value.shape[0]} samples, expected {self.sample_count}"
                )
            if not bool(torch.isfinite(value).all()):
                raise CaptureRunnerError(f"activation space {name!r} contains non-finite values")

    def require(self, name: str):
        try:
            return self.spaces[name]
        except KeyError as exc:
            raise CaptureRunnerError(f"missing activation space: {name}") from exc


@dataclass(frozen=True)
class PhiCaptureLayout:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @classmethod
    def from_model(cls, model: Any) -> "PhiCaptureLayout":
        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "phi3":
            raise CaptureRunnerError("Phi capture requires a Phi3ForCausalLM-style model")
        hidden = int(config.hidden_size)
        q_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", hidden // q_heads))
        if q_heads * head_dim != hidden:
            raise CaptureRunnerError("Phi Q projection width does not equal hidden_size")
        return cls(
            num_layers=int(config.num_hidden_layers),
            hidden_size=hidden,
            intermediate_size=int(config.intermediate_size),
            num_attention_heads=q_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
        )

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class GLM53CaptureLayout:
    num_layers: int
    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    mlp_layer_types: tuple[str, ...]

    @classmethod
    def from_model(cls, model: Any) -> "GLM53CaptureLayout":
        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "glm_moe_dsa":
            raise CaptureRunnerError(
                "GLM-5.3 capture requires a GlmMoeDsaForCausalLM-style model"
            )
        layer_types = tuple(
            str(x) for x in getattr(config, "mlp_layer_types", ())
        )
        if len(layer_types) != int(config.num_hidden_layers):
            first_dense = int(getattr(config, "first_k_dense_replace", 3))
            layer_types = tuple(
                "dense" if i < first_dense else "sparse"
                for i in range(int(config.num_hidden_layers))
            )
        return cls(
            num_layers=int(config.num_hidden_layers),
            hidden_size=int(config.hidden_size),
            q_lora_rank=int(config.q_lora_rank),
            kv_lora_rank=int(config.kv_lora_rank),
            qk_rope_head_dim=int(config.qk_rope_head_dim),
            mlp_layer_types=layer_types,
        )


def glm53_capture_taps(layout: GLM53CaptureLayout) -> tuple[ActivationTap, ...]:
    taps: list[ActivationTap] = [
        ActivationTap("embedding", "model.embed_tokens")
    ]
    for layer in range(layout.num_layers):
        prefix = f"model.layers.{layer}"
        name = f"layer.{layer}"
        taps.extend(
            [
                ActivationTap(
                    f"{name}.residual_in",
                    f"{prefix}.input_layernorm",
                    capture="input",
                ),
                ActivationTap(
                    f"{name}.attn_in",
                    f"{prefix}.input_layernorm",
                ),
                ActivationTap(
                    f"{name}.q_a",
                    f"{prefix}.self_attn.q_a_proj",
                ),
                ActivationTap(
                    f"{name}.q_resid",
                    f"{prefix}.self_attn.q_a_layernorm",
                ),
                ActivationTap(
                    f"{name}.q_b",
                    f"{prefix}.self_attn.q_b_proj",
                ),
                ActivationTap(
                    f"{name}.compressed_kv",
                    f"{prefix}.self_attn.kv_a_proj_with_mqa",
                ),
                ActivationTap(
                    f"{name}.attn_out",
                    f"{prefix}.self_attn",
                    tensor_index=0,
                ),
                ActivationTap(
                    f"{name}.mlp_in",
                    f"{prefix}.post_attention_layernorm",
                ),
                ActivationTap(
                    f"{name}.mlp_out",
                    f"{prefix}.mlp",
                ),
                ActivationTap(
                    f"{name}.residual_out",
                    prefix,
                    tensor_index=0,
                ),
            ]
        )
        if layout.mlp_layer_types[layer] == "sparse":
            taps.append(
                ActivationTap(
                    f"{name}.router_topk",
                    f"{prefix}.mlp.gate",
                    tensor_index=2,
                )
            )
    taps.append(ActivationTap("final", "model.norm"))
    return tuple(taps)


def iq_hybrid_residual_taps(num_layers: int) -> tuple[ActivationTap, ...]:
    if num_layers <= 0:
        raise CaptureRunnerError("num_layers must be positive")
    taps: list[ActivationTap] = [
        ActivationTap("embedding", "embed_tokens")
    ]
    for layer in range(num_layers):
        taps.append(
            ActivationTap(
                f"layer.{layer}.residual_in",
                f"layers.{layer}.norm",
                capture="input",
            )
        )
    taps.append(
        ActivationTap("final_in", "norm", capture="input")
    )
    taps.append(ActivationTap("final", "norm"))
    return tuple(taps)


def phi_capture_taps(num_layers: int) -> tuple[ActivationTap, ...]:
    if num_layers <= 0:
        raise CaptureRunnerError("num_layers must be positive")
    taps: list[ActivationTap] = [ActivationTap("embedding", "model.embed_tokens")]
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        name = f"layer.{layer}"
        taps.extend(
            [
                ActivationTap(f"{name}.attn_in", f"{prefix}.input_layernorm"),
                ActivationTap(f"{name}.qkv", f"{prefix}.self_attn.qkv_proj"),
                ActivationTap(f"{name}.attn_out", f"{prefix}.self_attn", tensor_index=0),
                ActivationTap(f"{name}.mlp_in", f"{prefix}.post_attention_layernorm"),
                ActivationTap(f"{name}.gate_up", f"{prefix}.mlp.gate_up_proj"),
                ActivationTap(f"{name}.mlp_out", f"{prefix}.mlp"),
                ActivationTap(f"{name}.residual_out", prefix),
            ]
        )
    taps.append(ActivationTap("final", "model.norm"))
    return tuple(taps)


def iq_capture_taps(num_layers: int) -> tuple[ActivationTap, ...]:
    if num_layers <= 0:
        raise CaptureRunnerError("num_layers must be positive")
    taps: list[ActivationTap] = [ActivationTap("embedding", "embed_tokens")]
    for layer in range(num_layers):
        prefix = f"blocks.{layer}"
        name = f"layer.{layer}"
        taps.extend(
            [
                ActivationTap(f"{name}.attn_in", f"{prefix}.input_norm"),
                ActivationTap(f"{name}.q", f"{prefix}.attn.q_proj"),
                ActivationTap(f"{name}.k", f"{prefix}.attn.k_proj"),
                ActivationTap(f"{name}.v", f"{prefix}.attn.v_proj"),
                ActivationTap(f"{name}.attn_out", f"{prefix}.attn"),
                ActivationTap(f"{name}.mlp_in", f"{prefix}.post_attention_norm"),
                ActivationTap(f"{name}.gate", f"{prefix}.mlp.gate_proj"),
                ActivationTap(f"{name}.up", f"{prefix}.mlp.up_proj"),
                ActivationTap(f"{name}.mlp_out", f"{prefix}.mlp"),
                ActivationTap(f"{name}.residual_out", prefix),
            ]
        )
    taps.append(ActivationTap("final", "norm"))
    return tuple(taps)


def _input_device(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.device
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise CaptureRunnerError("model has no parameters") from exc


def _prepare_batch(
    batch: Mapping[str, Any],
    device: Any,
    *,
    allow_padding: bool,
) -> tuple[dict[str, Any], Any]:
    torch = _require_torch()
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise CaptureRunnerError("each capture batch requires rank-2 input_ids")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise CaptureRunnerError("input_ids must be integer tensors")

    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise CaptureRunnerError("attention_mask must be a tensor with the same shape as input_ids")

    mask = attention_mask.to(dtype=torch.bool, device="cpu")
    if not allow_padding and not bool(mask.all()):
        raise CaptureRunnerError("IQ capture currently requires padding-free/packed batches")

    kwargs: dict[str, Any] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    if "position_ids" in batch:
        position_ids = batch["position_ids"]
        if not isinstance(position_ids, torch.Tensor) or position_ids.shape != input_ids.shape:
            raise CaptureRunnerError("position_ids must match input_ids shape")
        kwargs["position_ids"] = position_ids.to(device)
    if "document_ids" in batch:
        document_ids = batch["document_ids"]
        if not isinstance(document_ids, torch.Tensor) or document_ids.shape != input_ids.shape:
            raise CaptureRunnerError("document_ids must match input_ids shape")
        kwargs["document_ids"] = document_ids.to(device)
    return kwargs, mask


def _masked_concat(records: Mapping[str, tuple[Any, ...]], name: str, masks: list[Any]):
    torch = _require_torch()
    try:
        values = records[name]
    except KeyError as exc:
        raise CaptureRunnerError(f"capture record missing {name!r}") from exc
    if len(values) != len(masks):
        raise CaptureRunnerError(
            f"capture record {name!r} has {len(values)} batches, expected {len(masks)}"
        )

    selected: list[Any] = []
    for value, mask in zip(values, masks):
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise CaptureRunnerError(f"captured {name!r} must have shape [batch, sequence, features]")
        if tuple(value.shape[:2]) != tuple(mask.shape):
            raise CaptureRunnerError(f"captured {name!r} batch/sequence shape does not match attention mask")
        flat = value.float().reshape(-1, value.shape[-1])
        selected.append(flat[mask.reshape(-1)])
    result = torch.cat(selected, dim=0).contiguous()
    if result.numel() == 0:
        raise CaptureRunnerError(f"capture record {name!r} has no valid tokens")
    return result


def _masked_concat_router(
    records: Mapping[str, tuple[Any, ...]],
    name: str,
    masks: list[Any],
):
    torch = _require_torch()
    try:
        values = records[name]
    except KeyError as exc:
        raise CaptureRunnerError(f"capture record missing {name!r}") from exc
    if len(values) != len(masks):
        raise CaptureRunnerError(
            f"capture record {name!r} has {len(values)} batches, expected {len(masks)}"
        )
    selected: list[Any] = []
    for value, mask in zip(values, masks):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise CaptureRunnerError(
                f"captured {name!r} router indices must have shape [batch*sequence, top_k]"
            )
        tokens = int(mask.numel())
        if value.shape[0] != tokens:
            raise CaptureRunnerError(
                f"captured {name!r} router token count does not match attention mask"
            )
        reshaped = value.reshape(*mask.shape, value.shape[-1])
        selected.append(reshaped[mask])
    result = torch.cat(selected, dim=0).contiguous()
    if result.numel() == 0:
        raise CaptureRunnerError(f"capture record {name!r} has no valid tokens")
    if result.dtype not in (torch.int32, torch.int64):
        raise CaptureRunnerError(f"captured {name!r} router indices must be integer typed")
    return result


def _run_capture(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
    taps: tuple[ActivationTap, ...],
    *,
    phi: bool,
):
    torch = _require_torch()
    batch_list = list(batches)
    if not batch_list:
        raise CaptureRunnerError("at least one capture batch is required")
    device = _input_device(model)
    masks: list[Any] = []
    model.eval()
    with TorchActivationCapture(model, taps) as capture, torch.no_grad():
        for batch in batch_list:
            kwargs, mask = _prepare_batch(batch, device, allow_padding=True)
            masks.append(mask)
            if phi:
                if "document_ids" in kwargs:
                    raise CaptureRunnerError(
                        "Phi calibration batches must not use packed document_ids"
                    )
                model(**kwargs, use_cache=False)
            else:
                model(**kwargs)
    return capture.records(), masks


def capture_glm53_activations(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
) -> ActivationBundle:
    layout = GLM53CaptureLayout.from_model(model)
    batch_list = list(batches)
    if not batch_list:
        raise CaptureRunnerError("at least one capture batch is required")
    device = _input_device(model)
    masks: list[Any] = []
    torch = _require_torch()
    taps = glm53_capture_taps(layout)
    model.eval()
    with TorchActivationCapture(model, taps) as capture, torch.no_grad():
        for batch in batch_list:
            kwargs, mask = _prepare_batch(
                batch,
                device,
                allow_padding=True,
            )
            if "document_ids" in kwargs:
                raise CaptureRunnerError(
                    "GLM-5.3 calibration batches must not use packed document_ids"
                )
            masks.append(mask)
            model(**kwargs, use_cache=False)
    records = capture.records()
    spaces: dict[str, Any] = {
        "embedding": _masked_concat(records, "embedding", masks),
        "final": _masked_concat(records, "final", masks),
    }
    for layer in range(layout.num_layers):
        prefix = f"layer.{layer}"
        for suffix in (
            "residual_in",
            "attn_in",
            "q_a",
            "q_resid",
            "q_b",
            "compressed_kv",
            "attn_out",
            "mlp_in",
            "mlp_out",
            "residual_out",
        ):
            spaces[f"{prefix}.{suffix}"] = _masked_concat(
                records,
                f"{prefix}.{suffix}",
                masks,
            )
        if layout.mlp_layer_types[layer] == "sparse":
            spaces[f"{prefix}.router_topk"] = _masked_concat_router(
                records,
                f"{prefix}.router_topk",
                masks,
            )
    sample_count = int(spaces["embedding"].shape[0])
    return ActivationBundle(
        spaces=spaces,
        sample_count=sample_count,
        batch_count=len(masks),
    )


def capture_iq_hybrid_residuals(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
) -> ActivationBundle:
    config = getattr(model, "config", None)
    schedule = getattr(config, "schedule", None)
    if schedule is None or not hasattr(schedule, "layers"):
        raise CaptureRunnerError(
            "hybrid IQ capture requires IQHybridForCausalLM configuration"
        )
    batch_list = list(batches)
    if not batch_list:
        raise CaptureRunnerError("at least one capture batch is required")
    records, masks = _run_capture(
        model,
        batch_list,
        iq_hybrid_residual_taps(len(schedule.layers)),
        phi=False,
    )
    spaces: dict[str, Any] = {
        "embedding": _masked_concat(records, "embedding", masks),
        "final": _masked_concat(records, "final", masks),
        "final_in": _masked_concat(records, "final_in", masks),
    }
    for layer in range(len(schedule.layers)):
        spaces[f"layer.{layer}.residual_in"] = _masked_concat(
            records,
            f"layer.{layer}.residual_in",
            masks,
        )
        if layer + 1 < len(schedule.layers):
            spaces[f"layer.{layer}.residual_out"] = spaces[
                f"layer.{layer + 1}.residual_in"
            ]
        else:
            spaces[f"layer.{layer}.residual_out"] = spaces["final_in"]
    return ActivationBundle(
        spaces=spaces,
        sample_count=int(spaces["embedding"].shape[0]),
        batch_count=len(masks),
    )


def capture_phi_activations(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
) -> ActivationBundle:
    torch = _require_torch()
    layout = PhiCaptureLayout.from_model(model)
    records, masks = _run_capture(model, batches, phi_capture_taps(layout.num_layers), phi=True)
    spaces: dict[str, Any] = {
        "embedding": _masked_concat(records, "embedding", masks),
        "final": _masked_concat(records, "final", masks),
    }

    for layer in range(layout.num_layers):
        prefix = f"layer.{layer}"
        qkv = _masked_concat(records, f"{prefix}.qkv", masks)
        expected = layout.q_width + 2 * layout.kv_width
        if qkv.shape[1] != expected:
            raise CaptureRunnerError(
                f"Phi layer {layer} fused QKV width is {qkv.shape[1]}, expected {expected}"
            )
        q = qkv[:, : layout.q_width]
        k = qkv[:, layout.q_width : layout.q_width + layout.kv_width]
        v = qkv[:, layout.q_width + layout.kv_width :]

        gate_up = _masked_concat(records, f"{prefix}.gate_up", masks)
        if gate_up.shape[1] != 2 * layout.intermediate_size:
            raise CaptureRunnerError(
                f"Phi layer {layer} gate_up width is {gate_up.shape[1]}, "
                f"expected {2 * layout.intermediate_size}"
            )
        gate, up = gate_up.split(layout.intermediate_size, dim=-1)

        spaces.update(
            {
                f"{prefix}.attn_in": _masked_concat(records, f"{prefix}.attn_in", masks),
                f"{prefix}.q": q.contiguous(),
                f"{prefix}.k": k.contiguous(),
                f"{prefix}.v": v.contiguous(),
                f"{prefix}.attn_out": _masked_concat(records, f"{prefix}.attn_out", masks),
                f"{prefix}.mlp_in": _masked_concat(records, f"{prefix}.mlp_in", masks),
                f"{prefix}.mlp_gate": gate.contiguous(),
                f"{prefix}.mlp_up": up.contiguous(),
                f"{prefix}.mlp_hidden": (torch.nn.functional.silu(gate) * up).contiguous(),
                f"{prefix}.mlp_out": _masked_concat(records, f"{prefix}.mlp_out", masks),
                f"{prefix}.residual_out": _masked_concat(records, f"{prefix}.residual_out", masks),
            }
        )

    sample_count = int(spaces["embedding"].shape[0])
    return ActivationBundle(spaces=spaces, sample_count=sample_count, batch_count=len(masks))


def capture_iq_activations(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
) -> ActivationBundle:
    torch = _require_torch()
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "num_hidden_layers"):
        raise CaptureRunnerError("IQ capture requires a model with IQModelConfig-like config")

    records, masks = _run_capture(
        model,
        batches,
        iq_capture_taps(int(config.num_hidden_layers)),
        phi=False,
    )
    spaces: dict[str, Any] = {
        "embedding": _masked_concat(records, "embedding", masks),
        "final": _masked_concat(records, "final", masks),
    }

    for layer in range(int(config.num_hidden_layers)):
        prefix = f"layer.{layer}"
        gate = _masked_concat(records, f"{prefix}.gate", masks)
        up = _masked_concat(records, f"{prefix}.up", masks)
        spaces.update(
            {
                f"{prefix}.attn_in": _masked_concat(records, f"{prefix}.attn_in", masks),
                f"{prefix}.q": _masked_concat(records, f"{prefix}.q", masks),
                f"{prefix}.k": _masked_concat(records, f"{prefix}.k", masks),
                f"{prefix}.v": _masked_concat(records, f"{prefix}.v", masks),
                f"{prefix}.attn_out": _masked_concat(records, f"{prefix}.attn_out", masks),
                f"{prefix}.mlp_in": _masked_concat(records, f"{prefix}.mlp_in", masks),
                f"{prefix}.mlp_gate": gate,
                f"{prefix}.mlp_up": up,
                f"{prefix}.mlp_hidden": (torch.nn.functional.silu(gate) * up).contiguous(),
                f"{prefix}.mlp_out": _masked_concat(records, f"{prefix}.mlp_out", masks),
                f"{prefix}.residual_out": _masked_concat(records, f"{prefix}.residual_out", masks),
            }
        )

    sample_count = int(spaces["embedding"].shape[0])
    return ActivationBundle(spaces=spaces, sample_count=sample_count, batch_count=len(masks))


def make_activation_pair(
    source_fit: ActivationBundle,
    target_fit: ActivationBundle,
    source_validation: ActivationBundle,
    target_validation: ActivationBundle,
    *,
    source_space: str,
    target_space: str,
) -> ActivationPair:
    return ActivationPair(
        source_fit=source_fit.require(source_space),
        target_fit=target_fit.require(target_space),
        source_validation=source_validation.require(source_space),
        target_validation=target_validation.require(target_space),
        source_space=source_space,
        target_space=target_space,
    )


def build_phi_layer_calibration_from_bundles(
    source_fit: ActivationBundle,
    target_fit: ActivationBundle,
    source_validation: ActivationBundle,
    target_validation: ActivationBundle,
    *,
    source_layer: int,
    target_layer: int,
) -> PhiLayerCalibration:
    if source_layer < 0 or target_layer < 0:
        raise CaptureRunnerError("layer ids must be non-negative")
    source_prefix = f"layer.{source_layer}"
    target_prefix = f"layer.{target_layer}"

    def pair(suffix: str) -> ActivationPair:
        return make_activation_pair(
            source_fit,
            target_fit,
            source_validation,
            target_validation,
            source_space=f"{source_prefix}.{suffix}",
            target_space=f"{target_prefix}.{suffix}",
        )

    return PhiLayerCalibration(
        attn_in=pair("attn_in"),
        source_k_fit=source_fit.require(f"{source_prefix}.k"),
        attn_out=pair("attn_out"),
        mlp_in=pair("mlp_in"),
        mlp_hidden=pair("mlp_hidden"),
        mlp_out=pair("mlp_out"),
    )


def load_local_glm53_causal_lm(
    checkpoint: str | Path,
    *,
    device: str | None = None,
    dtype: str | Any = "bfloat16",
    attn_implementation: str = "sdpa",
):
    torch = _require_torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise CaptureRunnerError(
            "transformers is required to load GLM-5.3 checkpoints"
        ) from exc
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise CaptureRunnerError(
            f"GLM-5.3 checkpoint directory does not exist: {checkpoint}"
        )
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    resolved_dtype: Any = dtype
    if isinstance(dtype, str):
        if dtype == "auto":
            resolved_dtype = "auto"
        else:
            try:
                resolved_dtype = aliases[dtype.lower()]
            except KeyError as exc:
                raise CaptureRunnerError(
                    f"unsupported GLM-5.3 load dtype: {dtype}"
                ) from exc
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=resolved_dtype,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation=attn_implementation,
    )
    if getattr(model.config, "model_type", None) != "glm_moe_dsa":
        raise CaptureRunnerError(
            f"checkpoint model_type={getattr(model.config, 'model_type', None)!r} is not glm_moe_dsa"
        )
    if device is not None:
        model.to(torch.device(device))
    return model.eval()


def load_local_phi_causal_lm(
    checkpoint: str | Path,
    *,
    device: str | None = None,
    dtype: str | Any = "auto",
    attn_implementation: str = "sdpa",
):
    torch = _require_torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise CaptureRunnerError("transformers is required to load Phi checkpoints") from exc

    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise CaptureRunnerError(f"Phi checkpoint directory does not exist: {checkpoint}")

    resolved_dtype: Any = dtype
    if isinstance(dtype, str) and dtype != "auto":
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            resolved_dtype = aliases[dtype.lower()]
        except KeyError as exc:
            raise CaptureRunnerError(f"unsupported Phi load dtype: {dtype}") from exc

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=resolved_dtype,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation=attn_implementation,
    )
    if getattr(model.config, "model_type", None) != "phi3":
        raise CaptureRunnerError(
            f"checkpoint model_type={getattr(model.config, 'model_type', None)!r} is not Phi3/Phi-4"
        )
    if device is not None:
        model.to(torch.device(device))
    return model.eval()
