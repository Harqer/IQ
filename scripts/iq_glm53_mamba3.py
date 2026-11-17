#!/usr/bin/env python3
"""Load GLM-5.3 lexical tensors into native Mamba-3 MIMO and try GLM-token generation.

--glm-root must contain official GLM config/tokenizer files, a safetensors
index, and the two referenced lexical shards (not all expert shards).
This deliberately loads GLM's original embedding and untied LM head; its
Transformer/MoE reasoning must subsequently be distilled into the student.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from iq_model.glm_mamba3_student import (
    GLMMamba3Student, GLMMambaError, GLM_EOS_IDS, OFFICIAL_GEOMETRY,
)
from iq_transfer.mamba3_direct import (
    MAMBA3_MIMO_15B_REPO, MAMBA3_MIMO_15B_REVISION,
    MAMBA3_MIMO_15B_BIN_SHA256,
)

GLM_EMBED_KEY = "model.embed_tokens.weight"
GLM_HEAD_KEY = "lm_head.weight"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_config(root: Path) -> dict:
    path = root / "config.json"
    if not path.is_file():
        raise GLMMambaError("missing GLM-5.3 config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "glm_moe_dsa"
        or config.get("vocab_size") != OFFICIAL_GEOMETRY.glm_vocab
        or config.get("hidden_size") != OFFICIAL_GEOMETRY.glm_hidden
        or tuple(config.get("eos_token_id", ())) != GLM_EOS_IDS
    ):
        raise GLMMambaError("GLM-5.3 official lexical config mismatch")
    return config


def lexical_shard_paths(root: Path) -> dict[str, Path]:
    """Inspect only the index and refuse path traversal and missing tensors."""
    _source_config(root)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise GLMMambaError("missing official GLM safetensors index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise GLMMambaError("invalid GLM safetensors weight_map")
    results = {}
    for key in (GLM_EMBED_KEY, GLM_HEAD_KEY):
        relative = weight_map.get(key)
        if (not isinstance(relative, str) or not relative.endswith(".safetensors")
                or Path(relative).name != relative):
            raise GLMMambaError(f"invalid indexed lexical shard for {key}")
        path = root / relative
        if not path.is_file():
            raise GLMMambaError(f"missing GLM lexical shard: {relative}")
        results[key] = path
    return results


def load_glm_lexical_tensors(root: Path, device: torch.device, dtype: torch.dtype):
    """Stage only GLM embedding and head from their indexed safetensors shards."""
    from safetensors import safe_open
    shards = lexical_shard_paths(root)
    result = {}
    for key, path in shards.items():
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            if key not in stream.keys():
                raise GLMMambaError(f"index points to shard without {key}")
            value = stream.get_tensor(key)
        expected = (OFFICIAL_GEOMETRY.glm_vocab, OFFICIAL_GEOMETRY.glm_hidden)
        if tuple(value.shape) != expected or not value.is_floating_point():
            raise GLMMambaError(f"GLM {key} shape/dtype mismatch: {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise GLMMambaError(f"GLM {key} contains NaN/Inf")
        result[key] = value.to(device=device, dtype=dtype)
        del value
    return result[GLM_EMBED_KEY], result[GLM_HEAD_KEY]


def load_glm_student(
    root: Path,
    *,
    tokenizer_sha256: str | None,
    train_mamba_blocks: bool = False,
) -> GLMMamba3Student:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise GLMMambaError("native MIMO kernel requires a CUDA device supporting BF16")
    from scripts.patch_iq_native_mimo_tail import check_installed_kernel_guard
    check_installed_kernel_guard()
    from huggingface_hub import snapshot_download
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    _source_config(root)
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    snapshot = Path(snapshot_download(
        repo_id=MAMBA3_MIMO_15B_REPO,
        revision=MAMBA3_MIMO_15B_REVISION,
        allow_patterns=["config.json", "pytorch_model.bin"],
    ))
    weights = snapshot / "pytorch_model.bin"
    if not weights.is_file() or _hash_file(weights) != MAMBA3_MIMO_15B_BIN_SHA256:
        raise GLMMambaError("pretrained Mamba-3 weights are missing or changed")
    native = MambaLMHeadModel.from_pretrained(
        str(snapshot), device="cuda", dtype=torch.bfloat16,
    )
    embedding, head = load_glm_lexical_tensors(
        root, torch.device("cuda"), torch.bfloat16,
    )
    return GLMMamba3Student(
        native, embedding, head, tokenizer,
        expected_tokenizer_sha256=tokenizer_sha256,
        train_mamba_blocks=train_mamba_blocks,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-root", type=Path, help="Folder with GLM config/tokenizer/index/lexical shards")
    parser.add_argument("--glm-tokenizer-sha256",
                        help="Independently pinned SHA-256 of official token-to-ID mapping")
    parser.add_argument("--prompt", default="def square(x):\n    return")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--preflight", action="store_true",
                        help="Inspect source lexical shard coverage and device; do not load models")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_new_tokens > 128:
        parser.error("--max-new-tokens must be in 1..128")
    if not args.preflight and args.glm_root is None:
        parser.error("--glm-root is required for generation")
    result = {
        "architecture": "24 native Mamba-3 MIMO blocks + GLM-5.3 lexical adapters",
        "glm_vocab_size": OFFICIAL_GEOMETRY.glm_vocab,
        "glm_hidden": OFFICIAL_GEOMETRY.glm_hidden,
        "mamba_hidden": OFFICIAL_GEOMETRY.mamba_hidden,
        "native_mamba_donor_revision": MAMBA3_MIMO_15B_REVISION,
        "cuda": torch.cuda.is_available(),
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    if args.preflight:
        if args.glm_root is not None:
            result["lexical_shards"] = {
                key: str(path) for key, path in lexical_shard_paths(args.glm_root).items()
            }
        print(json.dumps(result, indent=2))
        return 0
    model = load_glm_student(
        args.glm_root, tokenizer_sha256=args.glm_tokenizer_sha256
    ).eval()
    result["student"] = model.generate_text(
        args.prompt, max_new_tokens=args.max_new_tokens,
    )
    result["glm_transformer_to_mamba_distillation"] = "not yet trained"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GLMMambaError, OSError, RuntimeError, ImportError, ValueError) as error:
        print(f"IQ_GLM_MAMBA3_FAILED: {error}", file=sys.stderr)
        sys.exit(1)
