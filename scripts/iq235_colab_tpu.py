"""Experimental full IQ235 source inference with one-layer-at-a-time TPU placement.

No CUDA kernels, expert quantization, checkpoint rewrite or reduced schedule.
Host owns depth residuals, lexical/reasoning modules and cached decode state.
The original 128256-vocabulary checkpoint is NOT a trained GLM-tokenized model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from tempfile import TemporaryDirectory
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
from iq_local.cache import LocalCache, checkpoint_cache_root, configure_reusable_budget
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

    def __init__(self, index, cache, client, *, max_ram_experts=0, prefetch=True):
        super().__init__(index, cache, client, max_ram_experts=max_ram_experts)
        self._io_lock = RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iq-prefetch") if prefetch and cache is not None else None
        self._pending = None
        self._base_files = {}
        for key, filename in index.weight_map.items():
            if key.startswith("layers.") and ".moe.routed_experts." not in key:
                pos = int(key.split(".")[1])
                self._base_files.setdefault(pos, set()).add(filename)
        self.counters.update(bucket_batches=0, prefetch_batches=0)

    def _ensure_shards(self, filenames, *, speculative=False):
        # Single cache writer per session. Unique staging directory per batch.
        with self._io_lock:
            missing = [name for name in sorted(set(filenames))
                       if not self.cache.contains(f"shards/{name}")]
            if not missing:
                return
            if speculative and self.cache.budget_bytes - self.cache.usage()[0] < 4 * (1 << 30):
                return
            for offset in range(0, len(missing), 8):
                batch = missing[offset:offset + 8]
                with TemporaryDirectory(prefix="iq-batch-", dir=self.cache.root) as staging:
                    paths = [(name, Path(staging) / name) for name in batch]
                    self.client.download(paths)
                    self.counters["bucket_batches"] += 1
                    if speculative:
                        self.counters["prefetch_batches"] += 1
                    for name, tmp in paths:
                        self.counters["bucket_files"] += 1
                        self.counters["bucket_bytes"] += tmp.stat().st_size
                        self.cache.commit(f"shards/{name}", tmp)

    def prefetch_layer(self, position):
        if self._pool is None or position not in self._base_files:
            return
        self.finish_prefetch()
        self._pending = self._pool.submit(
            self._ensure_shards, self._base_files[position], speculative=True)

    def finish_prefetch(self):
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.result()

    def close(self):
        if self._pool is not None:
            try:
                self.finish_prefetch()
            finally:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None

    def prepare_experts(self, stage, experts):
        self._ensure_shards([self._expert_filename(stage, int(e)) for e in experts])

    def load_layer_base(self, layer_idx):
        self.finish_prefetch()
        with self._io_lock:
            self._ensure_shards(self._base_files.get(layer_idx, ()))
            return super().load_layer_base(layer_idx)

    def load_global(self, key):
        with self._io_lock:
            return super().load_global(key)

    def _shard_path(self, filename):
        with self._io_lock:
            key = f"shards/{filename}"
            if self.cache.contains(key):
                self.counters["nvme_hits"] += 1
                return self.cache.touch(key)
            self._ensure_shards([filename])
            return self.cache.touch(key)

    def load_expert(self, stage, expert_idx):
        layer = self._expert_layer(stage)
        keys = [
            f"layers.{layer}.moe.routed_experts.{expert_idx}.{p}_proj.weight"
            for p in ("gate", "up", "down")
        ]
        filename = self._expert_filename(stage, expert_idx)
        if any(self.index.weight_map.get(k) != filename for k in keys):
            raise RuntimeError(f"Unexpected expert shard mapping: {stage}/{expert_idx}")
        with self._io_lock:
            tensors = self._read_keys(self._shard_path(filename), keys)
            return tuple(tensors[k] for k in keys)


class MountedStreamingProvider(RawStreamingProvider):
    """Read safetensors directly from a read-only HF mount; no duplicate cache."""

    def __init__(self, index, root):
        super().__init__(index, None, None, max_ram_experts=0)
        self.root = Path(root)
        self.counters["mounted_shard_opens"] = 0

    def prepare_experts(self, stage, experts):
        pass  # Mounted transport owns read-ahead.

    def load_layer_base(self, layer_idx):
        return StreamingWeightProvider.load_layer_base(self, layer_idx)

    def _shard_path(self, filename):
        if Path(filename).name != filename:
            raise ValueError(f"Invalid checkpoint filename: {filename}")
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Mounted checkpoint shard missing: {path}")
        self.counters["mounted_shard_opens"] += 1
        return path

    def _read_keys(self, path, keys):
        # Copy the requested tensors into host RAM while the mapping is open.
        # This avoids retaining remote mmap-backed storage across layer loads.
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key).clone() for key in keys}


def mounted_checkpoint_root(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path / RUN_PREFIX, path):
        if (candidate / "iq_config.json").is_file() and (
            candidate / "model.safetensors.index.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        f"No IQ checkpoint manifests under {path} or {path / RUN_PREFIX}. "
        "Check hf-mount status and the mounted bucket path."
    )


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


def moe_hidden(layer, x, expert_fn, prepare_experts=None):
    """Same routing and ascending expert accumulation; dynamic indices on host."""
    moe = layer.moe
    flat = layer.norm(x).reshape(-1, moe.config.hidden_size)
    _, selected, weights = moe._route(flat)
    selected_cpu = selected.cpu()
    weights_cpu = weights.cpu()
    latent = moe.latent_down(flat)
    routed = torch.zeros_like(latent)
    experts = sorted(set(selected_cpu.flatten().tolist()))
    if prepare_experts is not None:
        prepare_experts(moe._stage, experts)
    for expert in experts:
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
        # Construct real CPU buffers (e.g. RoPE) before moving to the native TPU.
        base = self.provider.load_layer_base(pos)
        materialize(layer, base, f"layers.{pos}.")
        return layer.to(self.device)

    def _run_layer(self, position, template, layer_input, *, prefill):
        layer = self._new_layer(position)
        self.provider.prefetch_layer(position + 1)
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
            out = moe_hidden(layer, x, self._expert_fn, self.provider.prepare_experts)
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
        # Use no_grad for accelerator compatibility; use the same streaming
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


class IQSession:
    """One initialized model/tokenizer; independent prompts share the weight cache."""
    def __init__(self, runner, provider, tokenizer, budget, mounted=False):
        self.runner, self.provider, self.tokenizer = runner, provider, tokenizer
        self.budget, self.mounted = budget, mounted
        self._closed = False
        self._request_lock = RLock()

    def ask(self, question, max_new_tokens=8):
        return self.complete(f"Question: {question}\nAnswer:", max_new_tokens)

    def complete(self, prompt, max_new_tokens=8):
        from scripts.iq_source_tokenizer import encode_prompt
        with self._request_lock:
            if self._closed or self.tokenizer is None:
                raise ValueError("A live text-mode session is required")
            print("Prompt:", prompt, flush=True)
            return self.generate_ids(encode_prompt(self.tokenizer, prompt), max_new_tokens)

    def generate_ids(self, prompt, max_new_tokens=2):
        with self._request_lock:
            if self._closed:
                raise RuntimeError("IQ session is closed")
            before = dict(self.provider.counters)
            output_ids = []
            tokenizer = self.tokenizer

            def report(event):
                print(json.dumps(event), flush=True)
                if event["phase"] == "token" and tokenizer is not None:
                    output_ids.append(event["token_id"])
                    print("IQ partial response:", tokenizer.decode(
                        output_ids, skip_special_tokens=True), flush=True)

            print(json.dumps({"mode": "experimental_source_inference",
                              "schedule": EXPECTED, "source_vocab": 128256,
                              "prompt_ids": prompt, "cache_gib": self.budget,
                              "weight_source": "mount" if self.mounted else "bucket_api"}), flush=True)
            start = time.monotonic()
            try:
                generated = self.runner.generate(
                    prompt, max_new_tokens, stop_tokens={128001, 128009} if tokenizer else None,
                    progress_callback=report)
            finally:
                self.provider.finish_prefetch()
            seconds = time.monotonic() - start
            result = {"generated_ids": generated, "seconds": seconds,
                      "tokens_per_second_including_prefill": len(generated) / seconds,
                      "provider": {k: v - before.get(k, 0) for k, v in self.provider.counters.items()},
                      "text": tokenizer.decode(generated, skip_special_tokens=True) if tokenizer else None}
            print(json.dumps(result), flush=True)
            if tokenizer is not None:
                print("IQ response:", result["text"], flush=True)
            return result

    def close(self):
        with self._request_lock:
            if not self._closed:
                self._closed = True
                try:
                    self.provider.close()
                finally:
                    self.runner = None
                    self.provider = None
                    self.tokenizer = None


def open_session(*, device="tpu", cache_dir=Path("/content/iq-cache"), cache_gib=None,
                 mount_dir=None, tokenizer_dir=None, official_tokenizer=False,
                 token_ids_only=False, prefetch=True):
    if device == "tpu":
        device = torch.device("tpu")
        print("Native TorchTPU:", device, "torch:", torch.__version__, flush=True)
        probe = torch.ones((8, 8), device=device, dtype=torch.bfloat16)
        result = float((probe @ probe).sum().cpu())
        if probe.device.type != "tpu" or result != 512.0:
            raise RuntimeError(f"Native TPU arithmetic failed: {probe.device}, {result}")
        del probe
        print("Native TPU BF16 matmul passed", flush=True)
    elif device == "cpu":
        device = torch.device("cpu")
        print("Explicit CPU reference mode", flush=True)
    else:
        raise ValueError("device must be tpu or cpu")
    tokenizer = None
    if not token_ids_only:
        from scripts.iq_source_tokenizer import load_source_tokenizer
        tokenizer = load_source_tokenizer(tokenizer_dir, official=official_tokenizer)
    if mount_dir is not None:
        root = mounted_checkpoint_root(mount_dir)
        print(f"Using mounted checkpoint: {root}", flush=True)
        index = RunIndex(None, root)
        config = validate_index(index)
        provider = MountedStreamingProvider(index, root)
        budget = None
    else:
        cache_root = checkpoint_cache_root(Path(cache_dir), BUCKET_ID, RUN_PREFIX)
        cache = LocalCache(cache_root / "raw", budget_bytes=0)
        budget = configure_reusable_budget(cache, cache_gib)
        print(json.dumps({"phase": "cache_open", "cached_bytes": cache.usage()[0],
                          "cached_shards": cache.usage()[1], "cache_gib": budget}), flush=True)
        client = BucketClient(BUCKET_ID, RUN_PREFIX)
        index = RunIndex(client, cache_root / "run")
        config = validate_index(index)
        provider = RawStreamingProvider(index, cache, client, prefetch=prefetch)
    try:
        model = build_host_model(config, provider)
        runner = TPUStreamingRunner(model, provider, device)
        return IQSession(runner, provider, tokenizer, budget, mount_dir is not None)
    except BaseException:
        provider.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--token-ids", nargs="+", type=int)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--official-tokenizer", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--mount-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/iq-cache"))
    parser.add_argument("--cache-gib", type=float, default=None)
    parser.add_argument("--device", choices=("tpu", "cpu"), default="tpu")
    parser.add_argument("--no-prefetch", action="store_true")
    args = parser.parse_args()
    session = open_session(device=args.device, cache_dir=args.cache_dir,
                           cache_gib=args.cache_gib, mount_dir=args.mount_dir,
                           tokenizer_dir=args.tokenizer_dir, official_tokenizer=args.official_tokenizer,
                           token_ids_only=args.token_ids is not None, prefetch=not args.no_prefetch)
    try:
        if args.token_ids is None:
            session.complete(args.prompt, args.max_new_tokens)
        else:
            session.generate_ids(args.token_ids, args.max_new_tokens)
    finally:
        session.close()


if __name__ == "__main__":
    main()
