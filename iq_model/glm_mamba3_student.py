"""GLM-5.3 lexical foundation on the *native* 24-block Mamba-3 MIMO model.

This is a cross-architecture student, not a relabelled tokenizer: its exact
GLM input/output weight tensors remain intact, with trainable width adapters.
The Mamba mixers and gated MLPs are the original pretrained modules.
GLM Transformer/MoE knowledge requires supervised teacher distillation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GLM_EOS_IDS = (154820, 154827, 154829)


class GLMMambaError(RuntimeError):
    """Incompatible architecture, tokenizer, weight identity, or token IDs."""


@dataclass(frozen=True)
class StudentGeometry:
    glm_vocab: int = 154880
    glm_hidden: int = 6144
    mamba_hidden: int = 2048
    mamba_layers: int = 24
    mimo_rank: int = 4
    state_size: int = 128
    max_context: int = 2048


OFFICIAL_GEOMETRY = StudentGeometry()


def tokenizer_mapping_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise GLMMambaError("GLM tokenizer must expose its full vocabulary")
    if any(type(key) is not str or type(value) is not int for key, value in vocab.items()):
        raise GLMMambaError("invalid GLM token-to-ID mapping")
    if len(set(vocab.values())) != len(vocab):
        raise GLMMambaError("GLM tokenizer contains duplicate IDs")
    blob = json.dumps(vocab, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def validate_geometry(
    source: nn.Module,
    token_embedding: Tensor,
    output_head: Tensor,
    tokenizer: Any,
    *,
    geometry: StudentGeometry = OFFICIAL_GEOMETRY,
    expected_tokenizer_sha256: str | None = None,
) -> str:
    if tuple(token_embedding.shape) != (geometry.glm_vocab, geometry.glm_hidden):
        raise GLMMambaError("GLM embedding shape mismatch")
    if tuple(output_head.shape) != (geometry.glm_vocab, geometry.glm_hidden):
        raise GLMMambaError("GLM LM-head shape mismatch")
    config = getattr(source, "config", None)
    backbone = getattr(source, "backbone", None)
    layers = getattr(backbone, "layers", None)
    if (
        getattr(config, "d_model", None) != geometry.mamba_hidden
        or getattr(config, "n_layer", None) != geometry.mamba_layers
        or not isinstance(layers, nn.ModuleList)
        or len(layers) != geometry.mamba_layers
    ):
        raise GLMMambaError("Mamba-3 donor geometry mismatch")
    ssm = getattr(config, "ssm_cfg", None)
    if not isinstance(ssm, dict) or (
        ssm.get("layer") != "Mamba3"
        or ssm.get("is_mimo") is not True
        or ssm.get("mimo_rank") != geometry.mimo_rank
        or ssm.get("d_state") != geometry.state_size
    ):
        raise GLMMambaError("donor must retain the native rank-4 MIMO recurrence")
    for i, block in enumerate(layers):
        mixer = getattr(block, "mixer", None)
        if (
            mixer is None or mixer.__class__.__name__ != "Mamba3"
            or getattr(mixer, "mimo_rank", None) != geometry.mimo_rank
            or getattr(mixer, "is_mimo", None) is not True
            or getattr(block.mlp, "__class__", type(None)).__name__ != "GatedMLP"
        ):
            raise GLMMambaError(f"native mixer or gated MLP missing at layer {i}")
    if getattr(source.backbone.embedding, "weight", None) is None:
        raise GLMMambaError("donor input embedding missing")
    if getattr(source.lm_head, "weight", None) is None:
        raise GLMMambaError("donor LM head missing")
    if source.backbone.embedding.weight.shape[1] != geometry.mamba_hidden:
        raise GLMMambaError("donor embedding hidden dimension mismatch")
    if source.lm_head.weight.shape[1] != geometry.mamba_hidden:
        raise GLMMambaError("donor head hidden dimension mismatch")
    parameters = tuple(source.parameters())
    if not parameters:
        raise GLMMambaError("empty donor")
    device, dtype = parameters[0].device, parameters[0].dtype
    if token_embedding.device != device or output_head.device != device:
        raise GLMMambaError("GLM lexical tensors and Mamba must share a device")
    if token_embedding.dtype != dtype or output_head.dtype != dtype:
        raise GLMMambaError("GLM lexical tensors and Mamba must share a dtype")
    if not token_embedding.is_floating_point() or not output_head.is_floating_point():
        raise GLMMambaError("GLM lexical tensors must be floating point")
    if (
        getattr(tokenizer, "eos_token_id", None) is None
        or not isinstance(getattr(tokenizer, "eos_token_id"), int)
        or not (0 <= tokenizer.eos_token_id < geometry.glm_vocab)
    ):
        raise GLMMambaError("invalid GLM EOS ID")
    if geometry == OFFICIAL_GEOMETRY and tokenizer.eos_token_id not in GLM_EOS_IDS:
        raise GLMMambaError("expected an official GLM EOS token")
    digest = tokenizer_mapping_hash(tokenizer)
    if any(not (0 <= token_id < geometry.glm_vocab)
           for token_id in tokenizer.get_vocab().values()):
        raise GLMMambaError("GLM tokenizer ID outside its embedding vocabulary")
    if expected_tokenizer_sha256 is not None:
        value = expected_tokenizer_sha256.lower().strip()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise GLMMambaError("tokenizer SHA-256 must be an independently pinned hash")
        if value != digest:
            raise GLMMambaError("GLM tokenizer mapping hash differs from pinned source")
    return digest


class GLMProjectedEmbedding(nn.Module):
    """Exact frozen GLM token vectors, followed by a learned 6144->2048 map."""

    def __init__(self, weight: Tensor, target_hidden: int):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.adapter = nn.Linear(weight.shape[1], target_hidden, bias=False,
                                 device=weight.device, dtype=weight.dtype)
        nn.init.normal_(self.adapter.weight, std=weight.shape[1] ** -0.5)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.adapter(F.embedding(input_ids, self.weight))


class GLMProjectedHead(nn.Module):
    """Learned 2048->6144 map then the *original*, untied GLM output head."""

    def __init__(self, weight: Tensor, source_hidden: int):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.adapter = nn.Linear(source_hidden, weight.shape[1], bias=False,
                                 device=weight.device, dtype=weight.dtype)
        nn.init.normal_(self.adapter.weight, std=source_hidden ** -0.5)

    def forward(self, hidden: Tensor) -> Tensor:
        return F.linear(self.adapter(hidden), self.weight)


class GLMMamba3Student(nn.Module):
    """Pretrained Mamba-3 MIMO backbone with GLM-native input/output vocabulary.

    Pass the original pinned MambaLMHeadModel and the TWO independently loaded
    GLM-5.3 lexical tensors. The donor source model is mutated in-place; do not
    reuse its reference after constructing this student.
    """

    def __init__(
        self,
        source_mamba: nn.Module,
        glm_embedding: Tensor,
        glm_lm_head: Tensor,
        glm_tokenizer: Any,
        *,
        geometry: StudentGeometry = OFFICIAL_GEOMETRY,
        expected_tokenizer_sha256: str | None = None,
        train_mamba_blocks: bool = False,
    ):
        super().__init__()
        self.mapping_sha256 = validate_geometry(
            source_mamba, glm_embedding, glm_lm_head, glm_tokenizer,
            geometry=geometry, expected_tokenizer_sha256=expected_tokenizer_sha256,
        )
        self.geometry = geometry
        self.tokenizer = glm_tokenizer
        self.tokenizer_mapping_pinned = expected_tokenizer_sha256 is not None
        # Freeze all original MIMO source weights unless explicitly fine-tuning.
        source_mamba.requires_grad_(False)
        source_mamba.backbone.embedding = GLMProjectedEmbedding(
            glm_embedding, geometry.mamba_hidden
        )
        source_mamba.lm_head = GLMProjectedHead(
            glm_lm_head, geometry.mamba_hidden
        )
        # The upstream config's tied embeddings flag is no longer valid.
        source_mamba.config.vocab_size = geometry.glm_vocab
        source_mamba.config.tie_embeddings = False
        if train_mamba_blocks:
            for block in source_mamba.backbone.layers:
                block.requires_grad_(True)
        self.core = source_mamba
        self.train_mamba_blocks = train_mamba_blocks

    def validate_input(self, input_ids: Tensor, new_tokens: int = 1) -> None:
        if (input_ids.ndim != 2 or input_ids.shape[1] == 0
                or input_ids.dtype not in (torch.int64, torch.int32)):
            raise GLMMambaError("input_ids must be nonempty integer [batch,sequence]")
        if new_tokens < 1 or input_ids.shape[1] + new_tokens > self.geometry.max_context:
            raise GLMMambaError("requested tokens exceed Mamba-3 context")
        if bool((input_ids < 0).any()) or bool((input_ids >= self.geometry.glm_vocab).any()):
            raise GLMMambaError("input contains IDs outside GLM vocabulary")

    def forward(self, input_ids: Tensor, **kwargs: Any) -> Any:
        self.validate_input(input_ids)
        return self.core(input_ids, **kwargs)

    @torch.inference_mode()
    def generate_text(self, prompt: str, *, max_new_tokens: int = 32) -> dict[str, Any]:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        self.validate_input(ids, max_new_tokens)
        ids = ids.to(self.core.backbone.embedding.weight.device)
        # Upstream Mamba's generator accepts one EOS integer, not a list.
        generated = self.core.generate(
            ids, max_length=ids.shape[1] + max_new_tokens,
            top_k=1, top_p=0.0, temperature=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        tail = generated[0, ids.shape[1]:].tolist()
        stop = next((i for i, token in enumerate(tail)
                     if token in GLM_EOS_IDS), len(tail))
        return {
            "completion": self.tokenizer.decode(tail[:stop], skip_special_tokens=True),
            "generated_tokens": len(tail),
            "glm_tokenizer_sha256": self.mapping_sha256,
            "tokenizer_pinned": self.tokenizer_mapping_pinned,
            "capability_transfer_trained": False,
        }
