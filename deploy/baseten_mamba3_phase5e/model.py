"""Custom Baseten Truss server for EXACT Phase 5e Mamba-3 MIMO weights.

The model is loaded ONCE per deployment. Each request returns real greedy
token IDs (not logits), and the client runs the entire evaluation suite.
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch

from iq_local.mamba3_csa_hca import (
    MODEL_ID, PHASE5E_SHA256, Mamba3CSAHCARunner,
)


class Model:
    def __init__(self, **kwargs):
        self.data_dir = Path(kwargs["data_dir"])
        self.lock = threading.Lock()
        self.runner = None

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Custom Mamba-3 Phase 5e Truss requires CUDA GPU")
        torch.set_num_threads(2)
        self.runner = Mamba3CSAHCARunner.from_checkpoints(
            self.data_dir / "iq_phase5e_attention_candidate.pt",
            expected_sha256=PHASE5E_SHA256,
            device="cuda",
            mode="hybrid",
        )

    def predict(self, model_input):
        if self.runner is None:
            raise RuntimeError("Verified Phase 5e runner has not loaded")
        if not isinstance(model_input, dict):
            raise ValueError("Expected a JSON object")
        mode = model_input.get("mode")
        ids = model_input.get("prompt_ids")
        count = model_input.get("max_new_tokens")
        stops = model_input.get("stop_tokens", [])
        if mode not in ("baseline", "hybrid"):
            raise ValueError("mode must be baseline or hybrid")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 2048
                or any(type(x) is not int or x < 0 or x >= 128256 for x in ids)):
            raise ValueError("Invalid original-model prompt token IDs")
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError("max_new_tokens must be 1..16")
        if len(ids) + count - 1 > 2048:
            raise ValueError("Context limit exceeded")
        if (not isinstance(stops, list)
                or any(type(x) is not int or x < 0 or x >= 128256 for x in stops)):
            raise ValueError("Invalid stop token IDs")
        # The two modes share weights, but each forward is explicitly selected.
        # Lock avoids accidental mode changes during concurrent requests.
        with self.lock:
            self.runner.mode = mode
            result = self.runner.generate(ids, count, stop_tokens=set(stops))
        return {
            "source_model": MODEL_ID,
            "checkpoint_sha256": PHASE5E_SHA256,
            "mode": mode,
            "generated_token_ids": result,
        }
