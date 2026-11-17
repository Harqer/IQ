"""Run all 24 real Mamba-3 MIMO blocks with GLM tokenizer IDs.

architecture_only: randomly initialized model; no pretrained neural weights.
injected: real Mamba backbone and GLM embedding/head tensors, same model class.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os

import torch

from iq_model.glm_mamba3_student import (
    GLMMamba3Student, OFFICIAL_GEOMETRY, GLM_EOS_IDS, tokenizer_mapping_hash,
)

GLM_REPO = "zai-org/GLM-5.3-BF16"
GLM_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAMBA_REPO = "state-spaces/mamba3-mimo-1.5b"
MAMBA_REVISION = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"


class PhaseError(RuntimeError):
    """Prevent invalid configurations and false-positive inference reports."""


def check_phase(phase):
    if phase not in ("architecture_only", "injected"):
        raise PhaseError("IQ_PHASE must be architecture_only or injected")
    return phase


def schema_digest(model):
    spec = sorted((name, list(t.shape), str(t.dtype))
                  for name, t in model.state_dict().items())
    return hashlib.sha256(json.dumps(spec, separators=(",", ":")).encode()).hexdigest()


def load_phase(phase, *, glm_root=None):
    check_phase(phase)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PhaseError("requires a BF16-capable CUDA GPU in Baseten")
    if tuple(torch.cuda.get_device_capability(0)) != (9, 0):
        raise PhaseError("pinned test requires Hopper SM90 / H100")
    if (torch.__version__.split("+", 1)[0] != "2.10.0"
            or torch.version.cuda != "12.8"):
        raise PhaseError("requires pinned PyTorch 2.10.0 and CUDA 12.8")
    from scripts.patch_iq_native_mimo_tail import check_installed_kernel_guard
    check_installed_kernel_guard()

    if phase == "injected":
        if not glm_root:
            raise PhaseError("injection requires IQ_GLM_ROOT with verified lexical shards")
        from scripts.iq_glm53_mamba3 import load_glm_student
        return load_glm_student(Path(glm_root), tokenizer_sha256=None).eval()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    conf = hf_hub_download(
        repo_id=MAMBA_REPO, filename="config.json", revision=MAMBA_REVISION)
    cfg = MambaConfig(**json.loads(Path(conf).read_text(encoding="utf-8")))
    if (cfg.n_layer != 24 or cfg.d_model != 2048 or cfg.vocab_size != 128256
            or cfg.ssm_cfg.get("layer") != "Mamba3"
            or cfg.ssm_cfg.get("is_mimo") is not True
            or cfg.ssm_cfg.get("mimo_rank") != 4
            or cfg.ssm_cfg.get("d_state") != 128):
        raise PhaseError("unexpected pinned Mamba architecture")
    tokenizer = AutoTokenizer.from_pretrained(
        GLM_REPO, revision=GLM_REVISION,
        use_fast=True, trust_remote_code=False)
    if (tokenizer.eos_token_id not in GLM_EOS_IDS
            or not 0 < len(tokenizer) <= OFFICIAL_GEOMETRY.glm_vocab):
        raise PhaseError("GLM tokenizer mismatch")
    torch.manual_seed(11726)
    device = torch.device("cuda:0")
    native = MambaLMHeadModel(cfg, device=device, dtype=torch.bfloat16)
    shape = (OFFICIAL_GEOMETRY.glm_vocab, OFFICIAL_GEOMETRY.glm_hidden)
    embedding = torch.empty(shape, device=device, dtype=torch.bfloat16)
    head = torch.empty(shape, device=device, dtype=torch.bfloat16)
    torch.nn.init.normal_(embedding, std=0.02)
    torch.nn.init.normal_(head, std=0.02)
    return GLMMamba3Student(native, embedding, head, tokenizer).eval()


@dataclass(frozen=True)
class ForwardReport:
    phase: str
    output: str
    generated_ids: list[int]
    input_tokens: int
    completed_layers: int
    schema_sha256: str
    tokenizer_sha256: str


@torch.inference_mode()
def full_forward_probe(model, prompt, *, phase, max_new_tokens=4, tracer=None):
    check_phase(phase)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        raise PhaseError("prompt must contain 1..4096 text characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 16:
        raise PhaseError("max_new_tokens must be 1..16")
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids
    model.validate_input(ids, new_tokens=max_new_tokens)
    n_input = int(ids.shape[1])
    ids = ids.to(model.core.backbone.embedding.weight.device)
    blocks = model.core.backbone.layers
    if len(blocks) != 24:
        raise PhaseError(f"expected 24 native Mamba blocks, got {len(blocks)}")
    hits = [0] * 24
    handles = []
    for i, block in enumerate(blocks):
        def visit(_mod, _args, _out, idx=i):
            hits[idx] += 1
            if tracer is not None:
                from opentelemetry import trace
                trace.get_current_span().add_event(
                    "iq.layer.completed", {"iq.layer.index": idx})
        handles.append(block.register_forward_hook(visit))
    generated = []
    try:
        for _ in range(max_new_tokens):
            ctx = tracer.start_as_current_span("iq.token.forward") if tracer else nullcontext(None)
            with ctx as span:
                previous = tuple(hits)
                logits = model(ids, num_last_tokens=1).logits[:, -1, :]
                if tuple(logits.shape) != (1, OFFICIAL_GEOMETRY.glm_vocab):
                    raise PhaseError(f"incorrect logits shape: {tuple(logits.shape)}")
                if not bool(torch.isfinite(logits).all()):
                    raise PhaseError("nonfinite output logits")
                if any(hits[i] != previous[i]+1 for i in range(24)):
                    raise PhaseError("some Mamba blocks did not execute")
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                val = int(next_id.item())
                generated.append(val)
                ids = torch.cat((ids, next_id.to(ids.device)), dim=-1)
                if span is not None:
                    span.set_attribute("iq.layers_completed", 24)
                    span.set_attribute("iq.tokens_total", int(ids.shape[1]))
                    if torch.cuda.is_available():
                        span.set_attribute("iq.cuda.memory_gib",
                            round(torch.cuda.memory_allocated(ids.device)/2**30,3))
            if val in GLM_EOS_IDS:
                break
    finally:
        for handle in handles:
            handle.remove()
    return ForwardReport(
        phase=phase, output=model.tokenizer.decode(generated, skip_special_tokens=True),
        generated_ids=generated, input_tokens=n_input, completed_layers=24,
        schema_sha256=schema_digest(model),
        tokenizer_sha256=tokenizer_mapping_hash(model.tokenizer),
    )
