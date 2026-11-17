"""Net-gated end-to-end smoke over the real bucket (Task 12).

Requires IQ_NET_TESTS=1 and HF_TOKEN exported; skipped otherwise, so the
offline suite stays green. Boots the Task 11 stack over the real
GLM-4.5-Air-derived run with a RAM-bounded provider (max_ram_experts=4,
mirroring a `--max-ram-experts 4` CLI), loads base weights through the
provider, generates one short completion for "The capital of France is"
with max_new_tokens=6, and records the cold-start first-token wall clock
plus provider counters for docs/IQ_LOCAL_STREAMING_VALIDATION.md.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.net

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 6


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    from iq_local.bucket import BUCKET_ID, RUN_PREFIX, BucketClient
    from iq_local.cache import LocalCache
    from iq_local.cpu_hybrid import CPUHybridForCausalLM
    from iq_local.index import RunIndex
    from iq_local.provider import StreamingWeightProvider
    from iq_local.runner import HybridRunner
    from transformers import AutoTokenizer

    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not exported", allow_module_level=False)
    work = tmp_path_factory.mktemp("smoke")
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    cache = LocalCache(work / "store", budget_bytes=8 << 30)
    index = RunIndex(client, work / "run")
    provider = StreamingWeightProvider(index, cache, client, max_ram_experts=4)
    config = index.config
    started = time.time()
    model = CPUHybridForCausalLM(config, dtype=torch.bfloat16, device="cpu")
    state: dict[str, torch.Tensor] = {}
    for key in index.weight_map:
        if not key.startswith("layers."):
            state[key] = provider.load_global(key)
    for layer_idx in range(len(config.schedule.layers)):
        state.update(provider.load_layer_base(layer_idx))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    model.eval()
    runner = HybridRunner(model, provider)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
    base_seconds = time.time() - started
    return {
        "runner": runner,
        "provider": provider,
        "tokenizer": tokenizer,
        "base_seconds": base_seconds,
    }


def test_real_bucket_streaming_smoke(stack):
    runner = stack["runner"]
    tokenizer = stack["tokenizer"]
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    prompt_ids = prompt_ids.reshape(-1).tolist()
    assert prompt_ids

    stop_ids = {tokenizer.eos_token_id}
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and 0 <= eot_id < len(tokenizer):
        stop_ids.add(eot_id)

    first_token_start = time.time()
    first = runner.generate(prompt_ids, 1)
    first_token_seconds = time.time() - first_token_start

    full_start = time.time()
    generated = runner.generate(prompt_ids, MAX_NEW_TOKENS, stop_tokens=stop_ids)
    full_seconds = time.time() - full_start

    assert len(first) == 1
    assert 1 <= len(generated) <= MAX_NEW_TOKENS
    text = tokenizer.decode(generated, skip_special_tokens=True)
    assert isinstance(text, str)

    counters = runner.provider.counters
    assert counters["bucket_files"] > 0
    bytes_used, entries = runner.provider.cache.usage()

    print(
        "\nIQ_SMOKE base_load_seconds=%.1f first_token_seconds=%.1f "
        "full_6_token_seconds=%.1f tokens_per_second=%.2f "
        "text=%r counters=%s cache_bytes=%d cache_entries=%d"
        % (
            stack["base_seconds"],
            first_token_seconds,
            full_seconds,
            len(generated) / full_seconds,
            text,
            counters,
            bytes_used,
            entries,
        ),
        flush=True,
    )
