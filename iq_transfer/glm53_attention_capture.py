"""Capture genuine GLM and IQ attention inputs with fast-tokenizer provenance.

The attention pre-hook stores exactly the tensor passed to the actual
attention module; it does NOT simulate activations or execute a surrogate
model. The caller runs the real model forward under this capture.

Hugging Face fast tokenizer return_offsets_mapping:
https://huggingface.co/docs/transformers/main_classes/tokenizer
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from torch import nn


class GLMAttentionCaptureError(RuntimeError):
    pass


def fast_tokenizer_text_provenance(
    tokenizer: Any,
    texts: list[str],
    *,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get tokenizer-native IDs, character offsets, exact raw UTF-8 hashes.

    Source and recipient may use different tokenizers. Neither may change
    the text, normalization choices, token input IDs or metadata after
    this call; the actual subsequent model-forward IDs are checked.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise GLMAttentionCaptureError(
            "character spans require a Hugging Face fast tokenizer"
        )
    if not texts or any(not isinstance(t, str) or not t for t in texts):
        raise GLMAttentionCaptureError("raw source text examples must be nonempty")
    try:
        encoding = tokenizer(
            texts, add_special_tokens=add_special_tokens,
            padding=False, truncation=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise GLMAttentionCaptureError("fast tokenizer offset mapping unavailable") from exc
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if not isinstance(ids, list) or not isinstance(offsets, list) or len(ids) != len(texts):
        raise GLMAttentionCaptureError("tokenizer result malformed")
    lengths = {len(row) for row in ids}
    if len(lengths) != 1 or any(len(off) != len(row) for row, off in zip(ids, offsets)):
        raise GLMAttentionCaptureError(
            "capture examples need equal token lengths; run separate unpadded microbatches"
        )
    id_tensor = torch.tensor(ids, dtype=torch.int64)
    offset_tensor = torch.tensor(offsets, dtype=torch.int64)
    hashes = torch.tensor(
        [list(sha256(text.encode("utf-8")).digest()) for text in texts],
        dtype=torch.uint8,
    )
    return id_tensor, offset_tensor, hashes


def resolve_attention_for_capture(
    model: nn.Module, *,
    stage: int,
    family: Literal["glm53", "iq"],
) -> nn.Module:
    """Find the actual donor or recipient layer rather than a sample module."""
    if type(stage) is not int or not 0 <= stage < 78:
        raise GLMAttentionCaptureError("stage must be 0..77")
    if family == "glm53":
        cfg = getattr(model, "config", None)
        if getattr(cfg, "model_type", None) != "glm_moe_dsa":
            raise GLMAttentionCaptureError(
                "donor must be an official glm_moe_dsa model"
            )
        network = getattr(model, "model", model)
        block = network.layers[stage]
        module = block.self_attn
        if not module.__class__.__module__.startswith(
            "transformers.models.glm_moe_dsa."
        ):
            raise GLMAttentionCaptureError(
                "donor attention must be official Transformers operator"
            )
    elif family == "iq":
        cfg = getattr(model, "config", None)
        if cfg is None or getattr(cfg.model, "num_hidden_layers", None) != 78:
            raise GLMAttentionCaptureError(
                "IQ capture requires the actual 78-stage IQ model"
            )
        block = model.layers[stage]
        module = block.attention
        from iq_model.attention.compressed import CompressedSparseContextAttention
        if not isinstance(module, CompressedSparseContextAttention):
            raise GLMAttentionCaptureError(
                "selected IQ stage is not compressed sparse attention"
            )
    else:
        raise GLMAttentionCaptureError("unsupported attention model family")
    return module


@contextmanager
def capture_attention_input(
    module: nn.Module,
    *,
    model_input_ids: torch.Tensor,
    tokenizer_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    text_sha256: torch.Tensor,
    kind: Literal["glm53", "iq"],
    output_file: str | Path,
) -> Iterator[None]:
    """Record attention *input* after caller performs one real model forward.

    Captures exactly once, validates all IDs against the tokenizer encoding,
    saves safetensors at exit, and always removes the pre-hook.
    """
    if kind not in ("glm53", "iq"):
        raise GLMAttentionCaptureError("unknown capture family")
    if (model_input_ids.shape != tokenizer_ids.shape or
        not torch.equal(model_input_ids.cpu(), tokenizer_ids.cpu())):
        raise GLMAttentionCaptureError(
            "model forward token IDs differ from tokenizer provenance"
        )
    if (token_offsets.shape != (*tokenizer_ids.shape, 2)
        or token_offsets.dtype not in (torch.int32, torch.int64)
        or text_sha256.dtype != torch.uint8
        or text_sha256.shape != (tokenizer_ids.shape[0], 32)):
        raise GLMAttentionCaptureError("invalid token span/text provenance")
    filename = Path(output_file)
    if filename.exists():
        raise GLMAttentionCaptureError("refuse to overwrite attention capture")
    captures: list[torch.Tensor] = []

    def on_input(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if len(captures) != 0:
            raise GLMAttentionCaptureError(
                "one capture context must cover exactly one model forward"
            )
        x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise GLMAttentionCaptureError("attention input must be a [B,S,H] tensor")
        if x.shape[:2] != tokenizer_ids.shape:
            raise GLMAttentionCaptureError("attention input token positions differ")
        if not bool(torch.isfinite(x).all()):
            raise GLMAttentionCaptureError("attention capture contains nonfinite values")
        captures.append(x.detach().cpu().contiguous())

    handle = module.register_forward_pre_hook(on_input, with_kwargs=True)
    try:
        yield
        if len(captures) != 1:
            raise GLMAttentionCaptureError("real attention forward was not observed")
        from safetensors.torch import save_file
        filename.parent.mkdir(parents=True, exist_ok=True)
        tensor_name = "source_layer_input" if kind == "glm53" else "iq_attention_input"
        save_file({
            tensor_name: captures[0],
            "token_ids": tokenizer_ids.detach().cpu().to(torch.int64).contiguous(),
            "token_offsets": token_offsets.detach().cpu().to(torch.int64).contiguous(),
            "text_sha256": text_sha256.detach().cpu().contiguous(),
        }, str(filename))
    finally:
        handle.remove()
