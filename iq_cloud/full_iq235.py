"""Exact 235-layer IQ phase-A inference, GLM tokenizer, no injected neural weights.

This is NOT the 24-layer Mamba student. A paid Baseten GPU run is separate
from the read-only preflight.
"""
from __future__ import annotations
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

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
    from .glm53_lexical import verify_glm53_tokenizer
    expected = os.getenv("IQ_GLM53_TOKENIZER_SHA256") or None
    return verify_glm53_tokenizer(
        tokenizer, source_config, expected_mapping_sha256=expected
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


def load_phase_a():
    if os.getenv("IQ_PHASE") != "architecture_only":
        raise FullIQ235Error("only architecture_only is implemented; injection follows the first stop")
    config_data = read_architecture_contract()
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


@torch.inference_mode()
def generate_probe(state, prompt, *, max_new_tokens=1, tracer=None):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1024:
        raise FullIQ235Error("prompt must be a nonempty string up to 1024 characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2:
        raise FullIQ235Error("phase A allows 1 or 2 complete forward tokens")
    ids = state.tokenizer(prompt, return_tensors="pt").input_ids
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 64:
        raise FullIQ235Error("phase A prompt must encode to 1..64 GLM token IDs")
    if bool(((ids < 0) | (ids >= EXPECTED_VOCAB)).any().item()):
        raise FullIQ235Error("GLM input token ID outside 154880 vocabulary")
    ids = ids.to("cuda:0")
    executed = [0 for _ in range(235)]
    hooks = []
    for index, layer in enumerate(state.model.layers):
        def on_forward(_module, _args, _output, idx=index):
            executed[idx] += 1
            if tracer is not None:
                from opentelemetry import trace
                trace.get_current_span().add_event("iq.physical_layer", {
                    "layer.index": idx, "layer.type": str(
                        state.model.config.schedule.layers[idx].value
                    ),
                })
        hooks.append(layer.register_forward_hook(on_forward))
    output_ids = []
    try:
        for step in range(max_new_tokens):
            scope = (tracer.start_as_current_span("iq235.full_forward")
                     if tracer else nullcontext())
            with scope:
                before = list(executed)
                prediction = state.model(ids)
                logits = prediction.logits[:, -1, :]
                if tuple(logits.shape) != (1, EXPECTED_VOCAB):
                    raise FullIQ235Error(f"wrong GLM vocabulary logits: {tuple(logits.shape)}")
                if not torch.isfinite(logits).all().item():
                    raise FullIQ235Error("nonfinite IQ logits")
                if any(executed[i] != before[i] + 1 for i in range(235)):
                    raise FullIQ235Error("one or more physical IQ layers failed to execute")
                token_id = int(logits.argmax(-1).item())
                if not 0 <= token_id < EXPECTED_VOCAB:
                    raise FullIQ235Error("GLM output token ID outside vocabulary")
                output_ids.append(token_id)
                if token_id in (154820, 154827, 154829):
                    break
                ids = torch.cat([ids, torch.tensor([[token_id]], device=ids.device)], -1)
    finally:
        for handle in hooks:
            handle.remove()
    return {
        "phase": "architecture_only",
        "checkpoint_weights_injected": False,
        "full_model": True,
        "physical_layers": 235,
        "mamba3_layers": 79,
        "context_layers": 78,
        "moe_layers": 78,
        "experts_per_moe_layer": 256,
        "glm_tokenizer_vocab": EXPECTED_VOCAB,
        "input_tokens": int(ids.shape[1] - len(output_ids)),
        "generated_ids": output_ids,
        "completion": state.tokenizer.decode(output_ids, skip_special_tokens=True),
        "schema_sha256": state.schema_sha256,
        "tokenizer_sha256": state.tokenizer_sha256,
        "all_layers_executed": True,
        "gpu_count": GPU_COUNT,
    }
