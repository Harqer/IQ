#!/usr/bin/env python3
"""Private HF Jobs chat server for the real GLM53 full78 IQ checkpoint.

Reuse the existing OpenAI-compatible /v1/chat/completions API, but load the
235-layer sparse-routed IQ model rather than the old full-resident mocha loader.
Keep non-expert weights on H200, retain selected expert weights in bounded
caches, prefill native Mamba-3 MIMO state once and decode the complete hybrid
backbone incrementally with CSA/HCA, dense-context, AttnRes and reasoning.

Serve only through HF Jobs authenticated proxy (expose port 8000 privately);
never enable public exposure. This script does not submit or cancel Jobs.
"""
from __future__ import annotations

from http.server import HTTPServer
import os
from pathlib import Path

import torch

from scripts import serve_iq_hf as existing_chat
from scripts.iq_hf_full78_conversation import load_runtime
from iq_model.hybrid_inference import IQHybridInferenceEngine
from scripts.verify_iq_mamba3_native_step import (verify_native_mimo_step, verify_full78_mimo_geometry)
from scripts.iq_hf_full78_sparse_inference import read_checkpoint

DEFAULT_CHECKPOINT = "/checkpoints/IQ/runs/glm53-full78-local-20261025a"


def main() -> None:
    # All 235 physical layers run during prefill and EVERY decode step.
    # Full-prefix recomputation is no longer the production inference path.
    existing_chat.MAX_PROMPT_TOKENS = 2048
    existing_chat.MAX_NEW_TOKENS = 256
    # Verify REAL pinned MIMO prefill/decode equivalence on the H200 before
    # spending minutes reading 315 baseline shards and thousands of experts.
    # A broken/unsupported recurrent kernel MUST fail closed, not quietly
    # route tokens through the old full-prefix implementation.
    verify_native_mimo_step(torch.device("cuda:0"))
    checkpoint = Path(os.getenv("IQ_CHECKPOINT_DIR", DEFAULT_CHECKPOINT))
    _, checkpoint_config, _ = read_checkpoint(checkpoint)
    verify_full78_mimo_geometry(torch.device("cuda:0"), checkpoint_config.mamba3)
    tokenizer, model, streamer = load_runtime(
        checkpoint,
        Path(os.getenv("IQ_TOKENIZER_DIR", "/tokenizer")),
        device=torch.device("cuda:0"),
    )
    port = int(os.getenv("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be a valid TCP port")
    engine = IQHybridInferenceEngine(model, max_context_tokens=2048)
    server = HTTPServer(("0.0.0.0", port), existing_chat.create_handler(
        tokenizer, model, engine=engine,
    ))
    print(
        "IQ_FULL78_PRIVATE_CHAT_READY="
        f"port={port} max_prompt_tokens=2048 max_new_tokens=256 "
        "decode_strategy=native_hybrid_incremental",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        streamer.close()


if __name__ == "__main__":
    main()
