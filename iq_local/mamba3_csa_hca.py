"""Experimental Mamba-3 MIMO + Phase 5e CSA/HCA CPU inference backend.

Separate from IQ's 235-layer GLM/MoE runner. The original 24-layer foundation
is frozen and the two residuals are loaded from the approved attention-only
checkpoint. Full-prefix CPU recomputation is slow; official CUDA MIMO parity
and general retrieval quality have NOT been established.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn

MODEL_ID = "state-spaces/mamba3-mimo-1.5b"
PHASE5E_SHA256 = "56cedd804224ca7f5db85834a8beddd461f19795f4b27b1b3503e11b4e2624f6"
PILOT_SHA256 = "f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53"
EXPERIMENT_FILE = (Path(__file__).resolve().parent.parent / "experiments" /
                   "mamba3-csa-hca" / "iq_mamba3_csa_hca_phase5e" / "iq_hybrid_cpu.py")


def _experiment():
    """Load the exact Phase 5e source without importing other phase modules."""
    if not EXPERIMENT_FILE.is_file():
        raise FileNotFoundError(f"Phase 5e experiment source missing: {EXPERIMENT_FILE}")
    name = "_iq_phase5e_inference_source"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXPERIMENT_FILE)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {EXPERIMENT_FILE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _foundation_hash(model: nn.Module) -> str:
    # Same deterministic name + FP32 bytes digest as Phase 5e's checksum().
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(memoryview(value.detach().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def _validate_metadata(checkpoint: dict) -> float:
    if checkpoint.get("source_model") != MODEL_ID:
        raise ValueError("Attention checkpoint does not target the original Mamba-3 MIMO model")
    if checkpoint.get("placement_after_blocks") != [8, 16]:
        raise ValueError("Attention insertion locations must be [8, 16]")
    if checkpoint.get("pilot_parent_sha256") != PILOT_SHA256:
        raise ValueError("Phase 5e checkpoint is not descended from the pinned Phase 4d pilot")
    if checkpoint.get("not_production_ready") is not True:
        raise ValueError("Unexpected Phase 5e checkpoint format")
    gain = float(checkpoint["max_relative_norm"])
    if not math.isfinite(gain) or not 0 < gain <= 0.1:
        raise ValueError("Invalid bounded-attention gain")
    if not isinstance(checkpoint.get("attention_state_dict"), dict):
        raise ValueError("Missing attention-only state dictionary")
    if not isinstance(checkpoint.get("source_model_sha256"), str):
        raise ValueError("Missing pretrained foundation hash")
    return gain


class BoundedResidual(nn.Module):
    """Identical per-token residual clipping rule to Phase 5e training."""
    def __init__(self, module: nn.Module, gain: float):
        super().__init__()
        self.module = module
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.module(x)
        xnorm = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dnorm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.gain * xnorm / dnorm).clamp(max=1)
        return delta * scale.to(delta.dtype)


class Mamba3CSAHCARunner:
    """Full-prefix greedy generation, compatible with iq_local.server.build_handler."""
    def __init__(self, foundation: nn.Module, attention: nn.ModuleDict, gain: float,
                 *, mode: str = "hybrid", context_limit: int = 2048):
        if mode not in ("hybrid", "baseline"):
            raise ValueError("mode must be 'hybrid' or 'baseline'")
        self.model = foundation.eval()
        self.attention = attention.eval()
        self.mode = mode
        self.vocab_size = int(foundation.cfg["vocab_size"])
        self.max_context_tokens = context_limit
        self.provider = None
        for p in foundation.parameters():
            p.requires_grad_(False)
        for p in attention.parameters():
            p.requires_grad_(False)
        self.insertions = nn.ModuleDict({key: BoundedResidual(module, gain)
                                        for key, module in attention.items()})

    @classmethod
    def from_checkpoints(cls, attention_path: Path, *, foundation_path: Path | None = None,
                         expected_sha256: str = PHASE5E_SHA256, mode: str = "hybrid"):
        attention_path = Path(attention_path).expanduser()
        if not attention_path.is_file():
            raise FileNotFoundError(f"Phase 5e checkpoint not found: {attention_path}")
        actual = _sha256(attention_path)
        if actual != expected_sha256:
            raise ValueError(f"Phase 5e SHA-256 mismatch: {actual} != {expected_sha256}")
        checkpoint = torch.load(attention_path, map_location="cpu", weights_only=True)
        gain = _validate_metadata(checkpoint)
        exp = _experiment()
        if foundation_path is None:
            from huggingface_hub import hf_hub_download
            foundation_path = Path(hf_hub_download(MODEL_ID, filename="pytorch_model.bin"))
        foundation_path = Path(foundation_path).expanduser()
        if not foundation_path.is_file():
            raise FileNotFoundError(f"Original Mamba-3 foundation not found: {foundation_path}")
        foundation, config = exp.load_cpu_checkpoint(foundation_path)
        actual_source_hash = _foundation_hash(foundation)
        if actual_source_hash != checkpoint["source_model_sha256"]:
            raise ValueError("Pretrained foundation parameter hash disagrees with Phase 5e")
        attention = exp.default_insertions(config)
        attention.load_state_dict(checkpoint["attention_state_dict"], strict=True)
        if not all(torch.isfinite(p).all().item() for p in attention.parameters()):
            raise ValueError("Attention checkpoint contains non-finite values")
        return cls(foundation, attention, gain, mode=mode)

    @torch.inference_mode()
    def next_logits(self, token_ids: list[int], *, baseline: bool | None = None) -> torch.Tensor:
        if not token_ids or len(token_ids) > self.max_context_tokens:
            raise ValueError(f"prompt must contain 1..{self.max_context_tokens} tokens")
        if any(type(token) is not int or token < 0 or token >= self.vocab_size for token in token_ids):
            raise ValueError("token IDs outside the pretrained Mamba-3 vocabulary")
        use_baseline = self.mode == "baseline" if baseline is None else baseline
        tokens = torch.tensor([token_ids], dtype=torch.long)
        logits = self.model(tokens, None if use_baseline else self.insertions)[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Mamba-3 inference returned non-finite logits")
        return logits

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int, *,
                 stop_tokens: set[int] | None = None, progress_callback=None) -> list[int]:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if len(prompt_ids) + max_new_tokens - 1 > self.max_context_tokens:
            raise ValueError("generation would exceed Mamba-3's validated context limit")
        prefix = list(prompt_ids)
        result = []
        for _ in range(max_new_tokens):
            next_token = int(self.next_logits(prefix).argmax())
            result.append(next_token)
            if progress_callback:
                progress_callback({"phase": "token", "token_index": len(result),
                                   "token_id": next_token})
            if stop_tokens and next_token in stop_tokens:
                break
            prefix.append(next_token)
        return result
