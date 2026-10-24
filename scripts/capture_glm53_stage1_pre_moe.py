#!/usr/bin/env python3
"""Capture real GLM-5.3-BF16 donor layer-3 PRE-MoE hidden states on CPU.

Stops after 4 official attention blocks + 3 dense MLPs, before sparse
routed experts or subsequent layers. Downloads only needed pinned tensor
byte ranges (some individual dense tensors are ~144 MiB).
Output is a new 2D .npy file suitable for --activations in the fidelity CLI.
This is not a full teacher inference or an IQ checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from iq_transfer.glm53_stream_capture import capture_glm53_bootstrap_streaming
from iq_transfer.http_range_source import HubSafetensorsSource
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text", action="append", required=True,
        help="text to tokenize; may repeat for several separate short examples",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="new .npy file of [tokens,6144] float32 input vectors",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="max tokens per text (1..256); recommended initial smoke: 16",
    )
    args = parser.parse_args()
    if (
        args.output.suffix != ".npy"
        or len(args.text) > 8
        or not 1 <= args.max_tokens <= 256
        or any(not item.strip() for item in args.text)
    ):
        parser.error(
            "require .npy output, 1..8 nonempty texts, and max-tokens 1..256"
        )
    sidecar = args.output.with_suffix(".json")
    if args.output.exists() or sidecar.exists():
        parser.error("output .npy and sidecar .json must not already exist")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            REPO, revision=REVISION, trust_remote_code=False
        )
        encoded = tokenizer(
            args.text, return_tensors="pt", truncation=True,
            max_length=args.max_tokens, padding=True,
        )
        ids = encoded["input_ids"].cpu()
        mask = encoded["attention_mask"].cpu()
        if int(mask.sum().item()) < 1:
            raise ValueError("tokenizer produced no valid tokens")
        config_file = Path(hf_hub_download(REPO, "config.json", revision=REVISION))
        index_file = Path(hf_hub_download(
            REPO, "model.safetensors.index.json", revision=REVISION
        ))
        # Metadata only in tmp: prevent accidental reuse of unverified local shards.
        with tempfile.TemporaryDirectory(prefix="glm53-moe-capture-index-") as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_bytes(index_file.read_bytes())
            source = HubSafetensorsSource(
                root, repo_id=REPO, revision=REVISION,
                max_tensor_bytes=192 * 1024 * 1024,
                max_request_bytes=4 * 1024 * 1024,
            )
            try:
                result = capture_glm53_bootstrap_streaming(
                    source=source,
                    config_path=config_file,
                    batches=[{"input_ids": ids, "attention_mask": mask}],
                    device="cpu",
                    attention_implementation="eager",
                    stop_at_pre_moe_layer=3,
                )
            finally:
                source.close()
        name = "layer.3.moe_input"
        if set(result.spaces) != {name}:
            raise ValueError("early capture returned unexpected activation spaces")
        values = result.require(name).detach().cpu().float().numpy()
        if values.ndim != 2 or values.shape[1] != 6144:
            raise ValueError("unexpected donor hidden width")
        if values.shape[0] != int(mask.sum().item()) or not np.isfinite(values).all():
            raise ValueError("captured token count or values are invalid")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "artifact_type": "glm53_real_pre_moe_hidden_capture",
            "not_an_iq_checkpoint": True,
            "model_repo": REPO,
            "model_revision": REVISION,
            "official_config_sha256": sha256_file(config_file),
            "official_index_sha256": sha256_file(index_file),
            "donor_layer": 3,
            "tap": name,
            "tap_position": "after post_attention_layernorm, before sparse MoE",
            "executed_layers": [0, 1, 2, 3],
            "routed_experts_executed": 0,
            "input_texts_count": len(args.text),
            "num_valid_tokens": int(values.shape[0]),
            "hidden_width": 6144,
            "dtype": "float32",
            "sha256": sha256_file(args.output),
            "token_ids_sha256": hashlib.sha256(
                ids.contiguous().numpy().tobytes()
            ).hexdigest(),
            "warning": (
                "Real donor hidden states, but not routed-expert-selected "
                "inputs, since MoE/router evaluation was intentionally skipped."
            ),
        }
        with sidecar.open("x", encoding="utf-8") as out:
            json.dump(metadata, out, indent=2, sort_keys=True)
            out.write("\n")
        print(json.dumps({
            "status": "real_pre_moe_hidden_states_captured",
            "tokens": metadata["num_valid_tokens"],
            "shape": list(values.shape),
            "output": str(args.output.resolve()),
            "sha256": metadata["sha256"],
            "not_full_model_inference": True,
        }, indent=2))
    except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        parser.exit(2, f"Pre-MoE capture failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
