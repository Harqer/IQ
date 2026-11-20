"""Full IQ235 architecture probe and separately guarded source-vocabulary experiment.

Architecture-only is the deployment default. The 128256-token pretrained
experiment is explicitly disabled until an independently pinned digest
manifest exists; neither path certifies language quality or production use.
"""
from __future__ import annotations
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from .inference_diagnostics import examine_layer_output, span_scope

import torch

EXPECTED_TYPES = {"mamba3": 79, "moe": 78, "csa": 21, "hca": 57}
EXPECTED_VOCAB = 154880
SOURCE_VOCAB = 128256
GLM_TOKENIZER = "zai-org/GLM-5.3-BF16"
GLM_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
GPU_COUNT = 8


class FullIQ235Error(RuntimeError):
    """Strict mismatch or incomplete execution of the real IQ backbone."""


def config_paths():
    root = Path(__file__).resolve().parents[1] / "configs"
    return (root / "iq235_checkpoint_source.json",
            root / "iq235_glm53_tokenizer_phase_a.json",
            root / "iq235_source_compile_status.json")


def read_architecture_contract():
    original_path, phase_a_path, status_path = config_paths()
    original = json.loads(original_path.read_text(encoding="utf-8"))
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    marker = json.loads(status_path.read_text(encoding="utf-8"))
    if original["model"]["vocab_size"] != SOURCE_VOCAB:
        raise FullIQ235Error("source checkpoint vocabulary has changed")
    if marker.get("inference_ready") is not False or marker.get("tokenizer_packaged") is not False:
        raise FullIQ235Error("unexpected source compile status")
    expected = json.loads(json.dumps(original))
    expected["model"]["vocab_size"] = EXPECTED_VOCAB
    if phase_a != expected:
        raise FullIQ235Error("phase A must change ONLY the tokenizer vocabulary")
    sequence = phase_a["schedule"]["layers"]
    if (len(sequence) != 235
            or phase_a["model"]["num_hidden_layers"] != 235
            or dict(Counter(sequence)) != EXPECTED_TYPES
            or phase_a["stable_moe"]["num_experts"] != 256
            or phase_a["moe"]["num_experts"] != 256
            or phase_a["mamba3"]["mimo_rank"] != 4
            or phase_a["mamba3"]["num_layers"] != 79
            or phase_a["model"]["hidden_size"] != 4096):
        raise FullIQ235Error("235-layer IQ physical topology mismatch")
    return phase_a


def physical_layer_placement(config, *, gpu_count=GPU_COUNT):
    """Round-robin by physical-layer index distributes 256-expert MoE stages."""
    if gpu_count != GPU_COUNT:
        raise FullIQ235Error("this Baseten contract requires exactly eight H200 GPUs")
    layers = config["schedule"]["layers"]
    if len(layers) != 235:
        raise FullIQ235Error("not the full IQ physical stack")
    ids = tuple(index % gpu_count for index in range(len(layers)))
    by_device = [Counter(layers[i] for i, gpu in enumerate(ids) if gpu == d)
                 for d in range(gpu_count)]
    if any(
        count.get("mamba3", 0) < 9
        or count.get("moe", 0) < 9
        or count.get("csa", 0) + count.get("hca", 0) < 9
        for count in by_device
    ):
        raise FullIQ235Error("imbalanced physical layer assignment")
    return ids


def tokenizer_contract(tokenizer, source_config):
    """Reuse the official GLM-5.3 lexical contract, never source IQ token IDs."""
    from .glm53_lexical import (
        GLM_PINNED_MAPPING_SHA256, verify_glm53_tokenizer,
    )
    supplied = os.getenv("IQ_GLM53_TOKENIZER_SHA256")
    if supplied and supplied.lower().strip() != GLM_PINNED_MAPPING_SHA256:
        raise FullIQ235Error("unverified override of pinned GLM tokenizer mapping")
    return verify_glm53_tokenizer(
        tokenizer, source_config,
        expected_mapping_sha256=GLM_PINNED_MAPPING_SHA256,
    )["mapping_sha256"]


def model_schema_sha(model):
    fields = sorted((name, tuple(t.shape), str(t.dtype))
                    for name, t in model.state_dict().items())
    data = json.dumps(fields, separators=(",", ":"))
    return sha256(data.encode("utf-8")).hexdigest()


@dataclass
class FullIQ235:
    tokenizer: object
    model: object
    schema_sha256: str
    tokenizer_sha256: str
    layer_devices: tuple[int, ...]
    phase: str = "architecture_only"
    checkpoint_weights_injected: bool = False
    verified_source_shards: int = 0


def load_phase_a():
    if os.getenv("IQ_PHASE") != "architecture_only":
        raise FullIQ235Error("only architecture_only is implemented; injection follows the first stop")
    config_data = read_architecture_contract()
    _require_worker_cuda()
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from iq_model.hybrid import IQHybridConfig, IQHybridForCausalLM
    from .glm53_lexical import inspect_lexical_boundary
    source_path = hf_hub_download(
        repo_id=GLM_TOKENIZER, filename="config.json", revision=GLM_REVISION,
        local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
    )
    source_config = json.loads(Path(source_path).read_text(encoding="utf-8"))
    inspect_lexical_boundary(
        json.loads(config_paths()[0].read_text(encoding="utf-8")), config_data
    )
    # No pretrained neural weights are loaded in this architecture-only phase.
    tokenizer = AutoTokenizer.from_pretrained(
        GLM_TOKENIZER, revision=GLM_REVISION, use_fast=True,
        trust_remote_code=False, local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
    )
    digest = tokenizer_contract(tokenizer, source_config)
    config = IQHybridConfig.from_dict(config_data)
    placement = physical_layer_placement(config_data)
    if config.model.vocab_size != EXPECTED_VOCAB:
        raise FullIQ235Error("config/tokenizer disagreement")
    devices = tuple(f"cuda:{i}" for i in placement)
    torch.manual_seed(23553)
    torch.cuda.manual_seed_all(23553)
    model = IQHybridForCausalLM(
        config, dtype=torch.bfloat16, device="cuda:0", layer_devices=devices
    ).eval()
    if len(model.layers) != 235:
        raise FullIQ235Error("incomplete constructed model")
    for i, module in enumerate(model.layers):
        parameters = list(module.parameters())
        if not parameters or any(p.device.index != placement[i] for p in parameters):
            raise FullIQ235Error(f"physical layer {i} has incorrect GPU placement")
    return FullIQ235(tokenizer, model, model_schema_sha(model), digest, placement)



def load_pretrained_source(*, tracer=None) -> FullIQ235:
    """Experimental full source-ID inference, not production quality certification.

    Strict gates run before constructing any GPU model or fetching weight data.
    No 539GB streaming is possible while the trusted digest pin remains None.
    """
    if os.getenv("IQ_PHASE") != "pretrained_source":
        raise FullIQ235Error("pretrained source mode requires explicit IQ_PHASE")
    from .checkpoint_metadata import (
        PINNED_SOURCE_SHARD_MANIFEST_SHA256, EXPECTED_SHARDS,
    )
    if PINNED_SOURCE_SHARD_MANIFEST_SHA256 is None:
        raise FullIQ235Error(
            "missing independently approved full source shard SHA-256 pin; "
            "no GPU or checkpoint loading authorized"
        )
    manifest_location = os.getenv("IQ235_APPROVED_MANIFEST_PATH")
    if not manifest_location:
        raise FullIQ235Error("approved manifest file path required for source mode")
    manifest_path = Path(manifest_location)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FullIQ235Error("approved manifest file missing or unsafe")
    if manifest_path.stat().st_size > 8_000_000:
        raise FullIQ235Error("approved manifest unexpectedly large")
    manifest = manifest_path.read_bytes()
    if sha256(manifest).hexdigest() != PINNED_SOURCE_SHARD_MANIFEST_SHA256:
        raise FullIQ235Error("approved source manifest digest mismatch")

    from transformers import AutoTokenizer
    from .lexical_warmstart import (
        LLAMA_PUBLIC_COPY, LLAMA_COPY_REVISION, LLAMA_COPY_MAPPING_SHA256,
        _mapping_hash,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_PUBLIC_COPY, revision=LLAMA_COPY_REVISION,
        use_fast=True, trust_remote_code=False,
        local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
    )
    digest = _mapping_hash(tokenizer, SOURCE_VOCAB)
    if digest != LLAMA_COPY_MAPPING_SHA256 or len(tokenizer) != SOURCE_VOCAB:
        raise FullIQ235Error("source Llama token-to-ID mapping is not pinned")

    # This still verifies the 235-layer schedule plus the GLM phase-A contract,
    # but the constructed source model uses the ORIGINAL 128256-token profile.
    read_architecture_contract()
    original = json.loads(config_paths()[0].read_text(encoding="utf-8"))
    _require_worker_cuda()
    from iq_model.hybrid import IQHybridConfig, IQHybridForCausalLM
    from .full235_weight_loader import load_verified_full235_body

    config = IQHybridConfig.from_dict(original)
    placement = physical_layer_placement(original)
    if config.model.vocab_size != SOURCE_VOCAB:
        raise FullIQ235Error("source model vocabulary geometry mismatch")
    with span_scope(tracer, "iq235.source.construct_full235", {
        "iq.physical_layers": 235, "iq.vocab_size": SOURCE_VOCAB,
    }):
        model = IQHybridForCausalLM(
            config, dtype=torch.bfloat16, device="cuda:0",
            layer_devices=tuple(f"cuda:{i}" for i in placement),
        ).eval()
    if len(model.layers) != 235:
        raise FullIQ235Error("source model must contain 235 physical layers")
    with span_scope(tracer, "iq235.source.install_verified_checkpoint", {
        "iq.shards.expected": EXPECTED_SHARDS,
    }):
        loaded = load_verified_full235_body(
            model, approved_manifest_bytes=manifest,
            approved_manifest_sha256=PINNED_SOURCE_SHARD_MANIFEST_SHA256,
            scratch_dir=os.getenv("IQ235_SHARD_SCRATCH_DIR"),
            install_source_lexical=True, tracer=tracer,
        )
    if (loaded.verified_source_shard_count != EXPECTED_SHARDS
            or loaded.loaded_lexical_tensor_count != 2
            or loaded.sampled_lexical_rows_verified != 7
            or not loaded.lexical_rows_initialized):
        raise FullIQ235Error("source lexical/checkpoint loading incomplete")
    # Only an experimental inference permit, NEVER a quality/production claim.
    return FullIQ235(
        tokenizer, model, model_schema_sha(model), digest, placement,
        phase="pretrained_source", checkpoint_weights_injected=True,
        verified_source_shards=loaded.verified_source_shard_count,
    )


def _require_worker_cuda() -> None:
    """Fail closed on any worker that is not the reviewed H200x8 topology."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != GPU_COUNT:
        raise FullIQ235Error("eight Baseten CUDA GPUs required; no CPU fallback")
    if (torch.__version__.split("+", 1)[0] != "2.10.0"
            or torch.version.cuda != "12.8"):
        raise FullIQ235Error("pinned CUDA PyTorch 2.10.0/12.8 environment required")
    for i in range(GPU_COUNT):
        if "H200" not in torch.cuda.get_device_name(i):
            raise FullIQ235Error(f"GPU {i} is not H200")
        if torch.cuda.get_device_capability(i) != (9, 0):
            raise FullIQ235Error(f"GPU {i} is not Hopper SM90")
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise FullIQ235Error(f"GPU {i} has no native BF16")
    from scripts.patch_iq_native_mimo_tail import check_installed_kernel_guard
    check_installed_kernel_guard()


@torch.inference_mode()
def generate_probe(state, prompt, *, max_new_tokens=1, tracer=None):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1024:
        raise FullIQ235Error("prompt must be a nonempty string up to 1024 characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2:
        raise FullIQ235Error("phase A allows 1 or 2 complete forward tokens")
    phase = getattr(state, "phase", "architecture_only")
    if phase not in {"architecture_only", "pretrained_source"}:
        raise FullIQ235Error("unexpected inference phase")
    if phase == "pretrained_source" and not state.checkpoint_weights_injected:
        raise FullIQ235Error("cannot infer before full pretrained source loading")
    vocab_size = SOURCE_VOCAB if phase == "pretrained_source" else EXPECTED_VOCAB
    diagnostics = os.getenv("IQ235_NUMERIC_DIAGNOSTICS") == "1"
    ids = state.tokenizer(prompt, return_tensors="pt").input_ids
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 64:
        raise FullIQ235Error("prompt must encode to 1..64 valid token IDs")
    if bool(((ids < 0) | (ids >= vocab_size)).any().item()):
        raise FullIQ235Error("input token ID outside model vocabulary")
    prompt_token_count = int(ids.shape[1])
    ids = ids.to("cuda:0")
    executed = [0 for _ in range(235)]
    hooks = []
    started_at: dict[int, float] = {}
    for index, layer in enumerate(state.model.layers):
        if diagnostics:
            def on_pre(_module, _args, idx=index):
                started_at[idx] = perf_counter()
            hooks.append(layer.register_forward_pre_hook(on_pre))

        def on_forward(_module, _args, _output, idx=index):
            executed[idx] += 1
            kind = str(state.model.config.schedule.layers[idx].value)
            current = None
            if tracer is not None:
                from opentelemetry import trace
                current = trace.get_current_span()
                current.add_event("iq.physical_layer", {
                    "layer.index": idx, "layer.type": kind,
                })
            if diagnostics:
                host_start = started_at.pop(idx, None)
                if host_start is not None and current is not None:
                    current.add_event("iq.layer.host_elapsed", {
                        "iq.layer.index": idx,
                        "iq.layer.host_elapsed_ms": (perf_counter() - host_start) * 1000,
                    })
                # CUDA synchronization makes this deliberately slow. It is
                # diagnostic host timing, NOT a GPU kernel latency measurement.
                examine_layer_output(
                    torch, _output, index=idx, layer_type=kind, span=current,
                )
        hooks.append(layer.register_forward_hook(on_forward))
    vocab = state.tokenizer.get_vocab()
    eos_ids = {state.tokenizer.eos_token_id} if (
        isinstance(state.tokenizer.eos_token_id, int)
    ) else set(state.tokenizer.eos_token_id or ())
    if phase == "pretrained_source":
        end_of_turn = vocab.get("<|eot_id|>")
        if isinstance(end_of_turn, int):
            eos_ids.add(end_of_turn)
        # The independent public Llama copies classify ID 128004 differently;
        # never let it appear as generated text in this experimental baseline.
        invalid_specials = set(state.tokenizer.all_special_ids) - eos_ids
        invalid_specials.add(128004)
        allowed_ids = set(vocab.values()) - invalid_specials
    else:
        allowed_ids = set(vocab.values())
    valid_output_ids = torch.tensor(sorted(allowed_ids), dtype=torch.long)
    if not len(valid_output_ids) or bool((valid_output_ids >= vocab_size).any()):
        raise FullIQ235Error("invalid token-to-ID output mapping")
    output_ids = []
    finish_reason = "length"
    try:
        for step in range(max_new_tokens):
            scope = (tracer.start_as_current_span("iq235.full_forward")
                     if tracer else nullcontext())
            with scope:
                before = list(executed)
                prediction = state.model(ids)
                logits = prediction.logits[:, -1, :]
                if tuple(logits.shape) != (1, vocab_size):
                    raise FullIQ235Error(f"wrong vocabulary logits: {tuple(logits.shape)}")
                if not torch.isfinite(logits).all().item():
                    raise FullIQ235Error("nonfinite IQ logits")
                if any(executed[i] != before[i] + 1 for i in range(235)):
                    raise FullIQ235Error("one or more physical IQ layers failed to execute")
                valid = valid_output_ids.to(device=logits.device)
                choice = int(logits[0].index_select(0, valid).argmax().item())
                token_id = int(valid_output_ids[choice].item())
                if not 0 <= token_id < vocab_size:
                    raise FullIQ235Error("output token ID outside vocabulary")
                output_ids.append(token_id)
                if token_id in (eos_ids if phase == "pretrained_source"
                                else (154820, 154827, 154829)):
                    finish_reason = "eos_token"
                    break
                ids = torch.cat([ids, torch.tensor([[token_id]], device=ids.device)], -1)
    finally:
        for handle in hooks:
            handle.remove()
    return {
        "phase": phase,
        "checkpoint_weights_injected": state.checkpoint_weights_injected,
        "full_model": True,
        "physical_layers": 235,
        "mamba3_layers": 79,
        "context_layers": 78,
        "moe_layers": 78,
        "experts_per_moe_layer": 256,
        "glm_tokenizer_vocab": (EXPECTED_VOCAB if phase == "architecture_only" else None),
        "source_tokenizer_vocab": (SOURCE_VOCAB if phase == "pretrained_source" else None),
        "verified_source_shards": state.verified_source_shards,
        "heldout_quality_verified": False,
        "input_tokens": prompt_token_count,
        "finish_reason": finish_reason,
        "generated_ids": output_ids,
        "completion": state.tokenizer.decode(output_ids, skip_special_tokens=True),
        "schema_sha256": state.schema_sha256,
        "tokenizer_sha256": state.tokenizer_sha256,
        "all_layers_executed": True,
        "gpu_count": GPU_COUNT,
    }
