"""Executable, unmodified pretrained Mamba-3 MIMO foundation inside IQ.

This is a deliberately separate inference lane from the experimental 235-layer
GLM/Mamba transplant. It preserves ALL 24 donor blocks, embeddings, norms,
recurrent state equations, and tied output head. No learned adapters are used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from iq_transfer.mamba3_direct import (
    MAMBA3_MIMO_15B_BIN_SHA256,
    MAMBA3_MIMO_15B_REPO,
    MAMBA3_MIMO_15B_REVISION,
)

OFFICIAL_TOKENIZER = "meta-llama/Llama-3.1-8B"
MAX_VALIDATED_CONTEXT = 2048


class FoundationInferenceError(RuntimeError):
    """Source artifact, architecture, or recurrent execution validation failed."""


def validate_native_contract(model: nn.Module, tokenizer: Any) -> None:
    """Inspect the REAL source model, not a reconstructed approximation."""
    config = getattr(model, "config", None)
    backbone = getattr(model, "backbone", None)
    layers = getattr(backbone, "layers", None)
    ssm = getattr(config, "ssm_cfg", None)
    expected = {
        "d_model": 2048,
        "d_intermediate": 3824,
        "n_layer": 24,
        "vocab_size": 128256,
        "tie_embeddings": True,
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": False,
    }
    if config is None or any(getattr(config, k, None) != v for k, v in expected.items()):
        raise FoundationInferenceError("official Mamba-3 1.5B model configuration mismatch")
    expected_ssm = {
        "layer": "Mamba3",
        "is_mimo": True,
        "mimo_rank": 4,
        "d_state": 128,
        "headdim": 64,
        "chunk_size": 16,
    }
    if not isinstance(ssm, dict) or any(ssm.get(k) != v for k, v in expected_ssm.items()):
        raise FoundationInferenceError("native recurrent configuration is not rank-4 Mamba-3 MIMO")
    if not isinstance(layers, nn.ModuleList) or len(layers) != 24:
        raise FoundationInferenceError("the complete ordered 24-block donor stack is required")
    for i, block in enumerate(layers):
        mixer = getattr(block, "mixer", None)
        mlp = getattr(block, "mlp", None)
        if (
            mixer is None or mlp is None
            or mixer.__class__.__name__ != "Mamba3"
            or mlp.__class__.__name__ != "GatedMLP"
            or getattr(mixer, "is_mimo", None) is not True
            or getattr(mixer, "mimo_rank", None) != 4
        ):
            raise FoundationInferenceError(f"missing native MIMO mixer or gated MLP at block {i}")
    embedding = getattr(getattr(backbone, "embedding", None), "weight", None)
    head = getattr(getattr(model, "lm_head", None), "weight", None)
    norm = getattr(getattr(backbone, "norm_f", None), "weight", None)
    if (
        embedding is None or head is None or norm is None
        or tuple(embedding.shape) != (128256, 2048)
        or tuple(head.shape) != (128256, 2048)
        or tuple(norm.shape) != (2048,)
        or embedding is not head
    ):
        raise FoundationInferenceError("pretrained embedding/head tying or global tensor geometry differs")
    if len(tokenizer) > 128256 or len(tokenizer) < 128000:
        raise FoundationInferenceError("tokenizer vocabulary is incompatible with pretrained embeddings")
    for token_id in (getattr(tokenizer, "bos_token_id", None),
                     getattr(tokenizer, "eos_token_id", None)):
        if token_id is not None and not 0 <= token_id < 128256:
            raise FoundationInferenceError("tokenizer special token is outside donor vocabulary")


def validate_ids(input_ids: torch.Tensor, max_new_tokens: int) -> None:
    if (
        input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0
        or input_ids.dtype not in (torch.int64, torch.int32)
    ):
        raise FoundationInferenceError("input_ids must be nonempty integer [1, sequence]")
    if max_new_tokens < 1 or input_ids.shape[1] + max_new_tokens > MAX_VALIDATED_CONTEXT:
        raise FoundationInferenceError("requested sequence exceeds source's validated 2048-token context")
    if bool((input_ids < 0).any()) or bool((input_ids >= 128256).any()):
        raise FoundationInferenceError("token IDs fall outside the official Mamba-3 vocabulary")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class IQMamba3Pretrained(nn.Module):
    """IQ-owned inference adapter around an unchanged full pretrained donor."""

    def __init__(self, source_model: nn.Module, tokenizer: Any):
        super().__init__()
        validate_native_contract(source_model, tokenizer)
        self.core = source_model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def load_official(cls, *, verify_hash: bool = True) -> "IQMamba3Pretrained":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise FoundationInferenceError(
                "native BF16 Mamba-3 MIMO requires a CUDA GPU with BF16 support; "
                "no simulated or SISO inference is substituted"
            )
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        snapshot = Path(snapshot_download(
            repo_id=MAMBA3_MIMO_15B_REPO,
            revision=MAMBA3_MIMO_15B_REVISION,
            allow_patterns=["config.json", "pytorch_model.bin"],
        ))
        weights = snapshot / "pytorch_model.bin"
        if not weights.is_file():
            raise FoundationInferenceError("official pretrained checkpoint was not downloaded")
        if verify_hash and _sha256_file(weights) != MAMBA3_MIMO_15B_BIN_SHA256:
            raise FoundationInferenceError("official pretrained checkpoint SHA-256 mismatch")
        tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_TOKENIZER, use_fast=True)
        source = MambaLMHeadModel.from_pretrained(
            str(snapshot), device="cuda", dtype=torch.bfloat16
        )
        return cls(source, tokenizer)

    def encode(self, prompt: str, max_new_tokens: int) -> torch.Tensor:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        validate_ids(ids, max_new_tokens)
        return ids.to(next(self.core.parameters()).device)

    @torch.inference_mode()
    def generate_text(self, prompt: str, *, max_new_tokens: int = 32) -> dict[str, Any]:
        ids = self.encode(prompt, max_new_tokens)
        torch.cuda.synchronize(ids.device)
        import time
        start = time.perf_counter()
        sequences = self.core.generate(
            ids, max_length=ids.shape[1] + max_new_tokens, top_k=1,
            top_p=0.0, temperature=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        torch.cuda.synchronize(ids.device)
        seconds = time.perf_counter() - start
        continuation = sequences[0, ids.shape[1]:]
        tokens = int(continuation.numel())
        return {
            "prompt": prompt,
            "completion": self.tokenizer.decode(continuation, skip_special_tokens=True),
            "prompt_tokens": int(ids.shape[1]),
            "generated_tokens": tokens,
            "wall_seconds": seconds,
            "end_to_end_tokens_per_second": tokens / seconds if seconds > 0 else None,
        }

    @torch.inference_mode()
    def check_cached_logits(self, prompt: str, *, atol: float = 0.25,
                            rtol: float = 0.025) -> dict[str, Any]:
        """Compare native cached prefill/one-token recurrent decoding with full forwards."""
        ids = self.encode(prompt, max_new_tokens=2)
        output = self.core.generate(
            ids, max_length=ids.shape[1] + 2, top_k=1,
            top_p=0.0, temperature=1.0, eos_token_id=None,
            return_dict_in_generate=True, output_scores=True,
        )
        scores = output.scores
        if len(scores) != 2 or output.sequences.shape[1] != ids.shape[1] + 2:
            raise FoundationInferenceError("native cached generator returned wrong token count")
        reports = []
        for step in range(2):
            prefix = output.sequences[:, : ids.shape[1] + step]
            full = self.core(prefix, num_last_tokens=1).logits[:, -1].float()
            cached = scores[step].float()
            if cached.shape != full.shape:
                raise FoundationInferenceError("cached logits changed vocabulary shape")
            delta = (full - cached).abs()
            ok = bool(torch.isfinite(full).all() and torch.isfinite(cached).all())
            passed = ok and bool(torch.allclose(full, cached, atol=atol, rtol=rtol))
            reports.append({
                "step": step,
                "max_abs_error": float(delta.max().item()),
                "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
                "allclose": passed,
            })
        if not all(r["allclose"] for r in reports):
            raise FoundationInferenceError(f"uncached/cached native MIMO parity failed: {reports}")
        return {"native_recurrence_parity": "passed", "steps": reports,
                "atol": atol, "rtol": rtol}
