"""Experimental full IQ235 source inference with one-layer-at-a-time TPU placement.

No CUDA kernels, expert quantization, checkpoint rewrite or reduced schedule.
Host owns depth residuals, lexical/reasoning modules and cached decode state.
The original 128256-vocabulary checkpoint is NOT a trained GLM-tokenized model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iq_local.bucket import BUCKET_ID, RUN_PREFIX, BucketClient
from iq_local.cache import LocalCache, checkpoint_cache_root
from iq_local.cpu_hybrid import (
    CPUHybridForCausalLM, CPUMamba3ResidualLayer,
    FunctionalStableLatentMoELayer, CompressedContextResidualLayer,
    DenseContextResidualLayer, RMSNorm, BlockAttentionResidual,
    ReasoningRecurrence, ReasoningStateInjector,
)
from iq_local.index import RunIndex
from iq_local.provider import StreamingWeightProvider
from iq_local.runner import HybridRunner
from iq_model.architecture import HybridLayerType
from iq_model.state.mamba3_cpu import CPUMamba3State

DTYPE = torch.bfloat16
SOURCE_TOKENIZER = "meta-llama/Llama-3.1-8B"
EXPECTED = {"mamba3": 79, "moe": 78, "csa": 21, "hca": 57}
SOURCE_CONFIG_SHA = "93f351e2bdd116ea5c83792cd1e494ce8dd904cb19c3bcac8c3cbecb1271a799"
SOURCE_INDEX_SHA = "0367412e31bb391a5c892497b565164a4b7954611250d1df1f0a986a8194b2bd"


class RawStreamingProvider(StreamingWeightProvider):
    """Bounded disk LRU for BOTH base and original BF16 expert shards."""

    def _shard_path(self, filename):
        key = f"shards/{filename}"
        if self.cache.contains(key):
            self.counters["nvme_hits"] += 1
            # Existing cache has no public touch method; refresh the LRU entry.
            self.cache._entries.move_to_end(key)
            return self.cache.path_for(key)
        return self._fetch_shard(filename, key)

    def load_expert(self, stage, expert_idx):
        layer = self._expert_layer(stage)
        keys = [
            f"layers.{layer}.moe.routed_experts.{expert_idx}.{p}_proj.weight"
            for p in ("gate", "up", "down")
        ]
        filename = self._expert_filename(stage, expert_idx)
        if any(self.index.weight_map.get(k) != filename for k in keys):
            raise RuntimeError(f"Unexpected expert shard mapping: {stage}/{expert_idx}")
        tensors = self._read_keys(self._shard_path(filename), keys)
        return tuple(tensors[k] for k in keys)


def validate_index(index):
    for name, expected in (
        ("iq_config.json", SOURCE_CONFIG_SHA),
        ("model.safetensors.index.json", SOURCE_INDEX_SHA),
    ):
        if sha256((index.work_dir / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected source checkpoint identity: {name}")
    config = index.config
    counts = Counter(x.value for x in config.schedule.layers)
    if counts != EXPECTED or config.model.vocab_size != 128256:
        raise RuntimeError(f"Unexpected architecture or vocabulary: {counts}")
    if config.mamba3.mimo_rank != 4 or config.stable_moe.top_k != 8:
        raise RuntimeError("Unexpected MIMO or MoE geometry")
    if len(index.weight_map) != 63164 or len(index.files) != 20283:
        raise RuntimeError("Incomplete IQ235 checkpoint inventory")
    for stage in range(78):
        expected = {f"model-moe-{stage:03d}-expert-{i:03d}.safetensors" for i in range(256)}
        if set(index.experts_by_stage.get(stage, [])) != expected:
            raise RuntimeError(f"Missing experts at stage {stage}")
    return config


def materialize(module, tensors, prefix):
    expected = module.state_dict()
    state = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
    if set(state) != set(expected):
        raise RuntimeError(f"{prefix}: missing={sorted(set(expected)-set(state))[:5]}, "
                           f"unexpected={sorted(set(state)-set(expected))[:5]}")
    for k, value in state.items():
        if value.shape != expected[k].shape:
            raise RuntimeError(f"{prefix}{k}: tensor shape mismatch")
    module.load_state_dict(state, strict=True)
    return module.eval()


def build_host_model(config, provider):
    # Meta skeleton describes all 235 layers without allocating their weights.
    model = CPUHybridForCausalLM(config, dtype=DTYPE, device="meta")
    expected = set(model.state_dict())
    indexed = {k for k in provider.index.weight_map if ".moe.routed_experts." not in k}
    if expected != indexed:
        raise RuntimeError(f"Base inventory mismatch: missing={sorted(expected-indexed)[:8]}, "
                           f"unexpected={sorted(indexed-expected)[:8]}")
    mc = config.model
    if mc.tie_word_embeddings:
        raise RuntimeError("This exact source runtime expects untied lexical matrices")
    factories = {
        "embed_tokens": lambda: nn.Embedding(mc.vocab_size, mc.hidden_size, dtype=DTYPE),
        "lm_head": lambda: nn.Linear(mc.hidden_size, mc.vocab_size, bias=False, dtype=DTYPE),
        "norm": lambda: RMSNorm(mc.hidden_size, mc.rms_norm_eps),
    }
    if config.attnres is not None:
        factories["attnres"] = lambda: BlockAttentionResidual(config.attnres)
    if config.reasoning is not None:
        factories["reasoning"] = lambda: ReasoningRecurrence(config.reasoning)
        factories["reasoning_injector"] = lambda: ReasoningStateInjector(
            mc.hidden_size, config.reasoning.state_dim)
    for name, factory in factories.items():
        prefix = name + "."
        module = factory().to(dtype=DTYPE)
        tensors = {k: provider.load_global(k) for k in indexed if k.startswith(prefix)}
        setattr(model, name, materialize(module, tensors, prefix))
    return model.eval()


def state_to(state, device):
    return CPUMamba3State(**{name: getattr(state, name).to(device)
                            for name in ("angle", "ssm", "k", "v")})


def moe_hidden(layer, x, expert_fn):
    """Same routing and ascending expert accumulation; dynamic indices on host."""
    moe = layer.moe
    flat = layer.norm(x).reshape(-1, moe.config.hidden_size)
    _, selected, weights = moe._route(flat)
    selected_cpu = selected.cpu()
    weights_cpu = weights.cpu()
    latent = moe.latent_down(flat)
    routed = torch.zeros_like(latent)
    for expert in sorted(set(selected_cpu.flatten().tolist())):
        rows, slots = torch.where(selected_cpu == expert)
        rows_dev = rows.to(x.device)
        result = moe._run_expert(expert_fn, expert, latent.index_select(0, rows_dev))
        w = weights_cpu[rows, slots].to(device=x.device, dtype=result.dtype)
        routed = routed.index_add(0, rows_dev, result * w.unsqueeze(-1))
    routed = moe.latent_up(moe.latent_norm(routed))
    shared = torch.zeros_like(flat)
    for expert in moe.shared_experts:
        shared = shared + expert(flat)
    return x + layer.residual_dropout((routed + shared).reshape(x.shape))


class TPUStreamingRunner(HybridRunner):
    def __init__(self, model, provider, device):
        super().__init__(model, provider)
        self.device = torch.device(device)
        self.mamba_indices = {}
        idx = 0
        for pos, typ in enumerate(self._schedule):
            if typ == HybridLayerType.MAMBA3:
                self.mamba_indices[pos] = idx
                idx += 1

    def _expert_fn(self, stage, expert):
        return tuple(w.to(device=self.device, dtype=DTYPE)
                     for w in self.provider.load_expert(stage, int(expert)))

    def _new_layer(self, pos):
        c = self.model.config
        typ = self._schedule[pos]
        if typ == HybridLayerType.MAMBA3:
            layer = CPUMamba3ResidualLayer(
                c.model, c.mamba3, mamba_layer_idx=self.mamba_indices[pos],
                dtype=DTYPE, device="cpu")
        elif typ == HybridLayerType.MOE:
            stage = self._moe_stage_of[pos]
            if self.provider.index.stage_of_layer(pos) != stage:
                raise RuntimeError(f"MoE stage mismatch at layer {pos}")
            layer = FunctionalStableLatentMoELayer(
                c.stable_moe, stage=stage, norm_eps=c.model.rms_norm_eps,
                residual_dropout=c.model.residual_dropout).to(dtype=DTYPE)
        elif typ in (HybridLayerType.CSA, HybridLayerType.HCA):
            layer = CompressedContextResidualLayer(
                c.model, c.compressed_context, mode=typ).to(dtype=DTYPE)
        else:
            raise RuntimeError(f"Unexpected layer {typ}")
        # Construct real CPU buffers (e.g. RoPE) before moving to XLA.
        base = self.provider.load_layer_base(pos)
        materialize(layer, base, f"layers.{pos}.")
        return layer.to(self.device)

    def _run_layer(self, position, template, layer_input, *, prefill):
        layer = self._new_layer(position)
        x = layer_input.to(self.device)
        if isinstance(layer, CPUMamba3ResidualLayer):
            # Explicit accelerator path through the plain-torch rank-4 core,
            # leaving the existing CPU-only public runtime contract untouched.
            normed = layer.norm(x)
            core = layer.mamba.core
            state = (CPUMamba3State.create(x.shape[0], layer.mamba.config,
                         device=self.device, kv_dtype=layer.mamba._kv_dtype)
                     if prefill else state_to(self._mamba_states[position], self.device))
            outputs = []
            for t in range(normed.shape[1]):
                y, state = core.step(normed[:, t], state)
                outputs.append(y)
            out = x + layer.residual_dropout(torch.stack(outputs, dim=1))
            self._mamba_states[position] = state_to(state, "cpu")
        elif isinstance(layer, FunctionalStableLatentMoELayer):
            out = moe_hidden(layer, x, self._expert_fn)
        elif isinstance(layer, (CompressedContextResidualLayer, DenseContextResidualLayer)):
            if not prefill:
                self._attn_cache[position] = [v.to(self.device)
                                              for v in self._attn_cache[position]]
            out = super()._run_layer(position, layer, x, prefill=prefill)
            self._attn_cache[position] = [v.cpu() for v in self._attn_cache[position]]
        else:
            raise RuntimeError(f"Unsupported layer {position}")
        # Materialize before dropping this layer's device parameters.
        result = out.cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError(f"Layer {position} produced nonfinite output")
        return result

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, *, stop_tokens=None, progress_callback=None):
        # Avoid inference_mode interaction with XLA; use the same streaming
        # prefill/decode composition as HybridRunner.
        if not prompt_ids or max_new_tokens < 1:
            raise ValueError("Prompt and positive generation length required")
        if any(not 0 <= i < self.model.model_config.vocab_size for i in prompt_ids):
            raise ValueError("Prompt token outside source vocabulary")
        self.reset()
        logits = self._prefill(prompt_ids, progress_callback=progress_callback)
        if progress_callback:
            self._notify_first_token_logits(progress_callback, logits)
        generated = []
        for step in range(max_new_tokens):
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite logits")
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if progress_callback:
                progress_callback({"phase": "token", "token_index": step+1, "token_id": token})
            if token in (stop_tokens or set()) or step+1 == max_new_tokens:
                break
            logits = self._decode_step(token, progress_callback=progress_callback)
        return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--token-ids", nargs="+", type=int,
                        help="Run real source-token inference without downloading a tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/iq-cache"))
    parser.add_argument("--cache-gib", type=float, default=None)
    parser.add_argument("--device", choices=("tpu", "cpu"), default="tpu")
    args = parser.parse_args()
    if args.device == "tpu":
        import torch_xla
        import torch_xla.runtime as xr
        xr.set_device_type("TPU")
        device = torch_xla.device()
        if xr.device_type() != "TPU":
            raise RuntimeError("No TPU backend; CPU fallback is not automatic")
        print("TPU:", device, "devices:", xr.global_runtime_device_count(), flush=True)
        # Prove device placement with real arithmetic, not just an import.
        probe = torch.ones((8, 8), device=device)
        assert float((probe @ probe).sum().cpu()) == 512.0
    else:
        device = torch.device("cpu")
        print("Explicit CPU reference mode", flush=True)
    tokenizer = None
    if args.token_ids is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(SOURCE_TOKENIZER)
        expected = {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
                    "<|start_header_id|>": 128006, "<|end_header_id|>": 128007,
                    "<|eot_id|>": 128009}
        if len(tokenizer) != 128256 or any(
            tokenizer.convert_tokens_to_ids(k) != v for k, v in expected.items()
        ):
            raise RuntimeError("Source tokenizer contract mismatch")
        prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    else:
        prompt = args.token_ids
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.cache_dir).free / (1 << 30)
    budget = args.cache_gib if args.cache_gib is not None else free_gib - 16
    if budget <= 0 or budget > free_gib - 12:
        raise RuntimeError("Insufficient scratch disk for requested cache budget")
    cache_root = checkpoint_cache_root(args.cache_dir, BUCKET_ID, RUN_PREFIX)
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    cache = LocalCache(cache_root / "raw", budget_bytes=int(budget * (1 << 30)))
    index = RunIndex(client, cache_root / "run")
    config = validate_index(index)
    provider = RawStreamingProvider(index, cache, client, max_ram_experts=0)
    model = build_host_model(config, provider)
    runner = TPUStreamingRunner(model, provider, device)
    def report(event):
        print(json.dumps(event), flush=True)
        if event["phase"] == "token" and tokenizer is not None:
            print("token:", repr(tokenizer.decode([event["token_id"]])), flush=True)
    print(json.dumps({"mode": "experimental_source_inference",
                      "schedule": EXPECTED, "source_vocab": 128256,
                      "prompt_ids": prompt, "cache_gib": budget}), flush=True)
    start = time.monotonic()
    stop = {tokenizer.eos_token_id, 128009} if tokenizer is not None else None
    generated = runner.generate(prompt, args.max_new_tokens,
                                stop_tokens=stop, progress_callback=report)
    seconds = time.monotonic() - start
    print(json.dumps({"generated_ids": generated, "seconds": seconds,
                      "tokens_per_second_including_prefill": len(generated)/seconds,
                      "provider": provider.counters,
                      "text": tokenizer.decode(generated) if tokenizer else None}), flush=True)


if __name__ == "__main__":
    main()
