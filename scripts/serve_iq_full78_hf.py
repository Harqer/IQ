#!/usr/bin/env python3
"""Private HF Jobs chat server for the real GLM53 full78 IQ checkpoint.

Reuse the existing OpenAI-compatible /v1/chat/completions API, but load the
235-layer sparse-routed IQ model rather than the old full-resident mocha loader.
Keep non-expert weights on H200, page selected expert weights for each forward,
and recompute the full prefix using Mamba-3 MIMO.

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

DEFAULT_CHECKPOINT = "/checkpoints/IQ/runs/glm53-full78-local-20261025a"


def main() -> None:
    # Full 235-layer model: expose ordinary multi-turn chat budgets, not the
    # original first-token smoke-test limits. Full-prefix decoding remains slow.
    existing_chat.MAX_PROMPT_TOKENS = 2048
    existing_chat.MAX_NEW_TOKENS = 256
    tokenizer, model, streamer = load_runtime(
        Path(os.getenv("IQ_CHECKPOINT_DIR", DEFAULT_CHECKPOINT)),
        Path(os.getenv("IQ_TOKENIZER_DIR", "/tokenizer")),
        device=torch.device("cuda:0"),
    )
    port = int(os.getenv("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be a valid TCP port")
    server = HTTPServer(("0.0.0.0", port), existing_chat.create_handler(tokenizer, model))
    print(
        "IQ_FULL78_PRIVATE_CHAT_READY="
        f"port={port} max_prompt_tokens=2048 max_new_tokens=256",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        streamer.close()


if __name__ == "__main__":
    main()
