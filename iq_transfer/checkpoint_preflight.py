"""Offline, allocation-free validation of IQ inference checkpoint metadata.

This checks filesystem/index integrity *before* initializing the costly
CUDA model and can be used by a local CI job with synthetic test fixtures.
It does not prove training quality, file content hashes or weight parity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CheckpointPreflightError(ValueError):
    pass


@dataclass(frozen=True)
class IQCheckpointPreflight:
    root: Path
    config: Mapping[str, Any]
    manifest: Mapping[str, Any] | None
    weight_count: int
    shard_count: int
    shard_bytes: int


def _object(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise CheckpointPreflightError(f"invalid JSON metadata: {path.name}") from exc
    if not isinstance(obj, dict):
        raise CheckpointPreflightError(f"{path.name} must contain a JSON object")
    return obj


def _shard_path(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise CheckpointPreflightError("checkpoint shard name must be a nonempty string")
    relative = Path(name)
    if (
        relative.name != name
        or relative.is_absolute()
        or relative.suffix != ".safetensors"
        or name in (".", "..")
        or "\\" in name
    ):
        raise CheckpointPreflightError(f"unsafe checkpoint shard path: {name!r}")
    shard = root / relative
    if shard.is_symlink() or not shard.is_file():
        raise CheckpointPreflightError(f"missing/symlink checkpoint shard: {name}")
    if shard.resolve().parent != root:
        raise CheckpointPreflightError(f"checkpoint shard escapes its directory: {name}")
    if shard.stat().st_size <= 0:
        raise CheckpointPreflightError(f"empty checkpoint shard: {name}")
    return shard


def preflight_iq_checkpoint(
    root: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
) -> IQCheckpointPreflight:
    """Reject partial/unsafe checkpoint manifests before CUDA allocations."""
    try:
        path = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CheckpointPreflightError("checkpoint root unavailable") from exc
    if not path.is_dir():
        raise CheckpointPreflightError("checkpoint root must be a directory")
    config = _object(path / "iq_config.json")
    model = config.get("model")
    if not isinstance(model, dict):
        raise CheckpointPreflightError("IQ config missing model settings")
    for key in ("vocab_size", "hidden_size"):
        if type(model.get(key)) is not int or model[key] <= 0:
            raise CheckpointPreflightError(f"invalid IQ model config field: {key}")
    schedule = config.get("schedule")
    if not isinstance(schedule, dict) or not isinstance(schedule.get("layers"), list):
        raise CheckpointPreflightError("missing IQ physical hybrid layer schedule")

    index = _object(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointPreflightError("checkpoint index has no weight_map")
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise CheckpointPreflightError("checkpoint index has malformed tensor keys")

    if any(not isinstance(name, str) or not name for name in weight_map.values()):
        raise CheckpointPreflightError("checkpoint index contains invalid shard names")
    names = set(weight_map.values())
    sizes = [_shard_path(path, name).stat().st_size for name in names]
    total = sum(sizes)
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata:
        declared_size = metadata["total_size"]
        if type(declared_size) is not int or declared_size != total:
            raise CheckpointPreflightError("index declared total_size mismatches shard bytes")

    manifest_path = path / "transplant_manifest.json"
    manifest = _object(manifest_path) if manifest_path.is_file() else None
    if manifest is not None:
        if "schema_version" in manifest and type(manifest["schema_version"]) is not int:
            raise CheckpointPreflightError("invalid transplant schema_version")
        if not isinstance(manifest.get("method"), str):
            raise CheckpointPreflightError("transplant manifest has no method")

    if tokenizer_dir is not None:
        source = Path(tokenizer_dir).resolve(strict=True)
        if not source.is_dir() or not any(
            (source / name).is_file()
            for name in ("tokenizer.json", "tokenizer.model")
        ):
            raise CheckpointPreflightError("tokenizer assets unavailable")
    return IQCheckpointPreflight(
        root=path,
        config=config,
        manifest=manifest,
        weight_count=len(weight_map),
        shard_count=len(names),
        shard_bytes=total,
    )


def verify_iq_tokenizer(
    tokenizer: Any,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> str:
    """Fail closed on a tokenizer incompatible with the recorded lexical donor.

    A compatible vocab size by itself is insufficient proof of token/ID
    mapping parity. Recognized donor-specific special IDs must also match.
    """
    model = config.get("model", {})
    size = model.get("vocab_size")
    if type(size) is not int or len(tokenizer) != size:
        raise CheckpointPreflightError("tokenizer vocabulary size differs from checkpoint")
    if manifest is None:
        raise CheckpointPreflightError(
            "missing transplant manifest: cannot validate which pretrained tokenizer owns the lexical weights"
        )
    method = manifest.get("method")
    if method == "mamba3_foundation_plus_glm53_capability_weight_transfer":
        foundation = manifest.get("foundation", {})
        if not isinstance(foundation, dict) or foundation.get("repo") != "state-spaces/mamba3-mimo-1.5b":
            raise CheckpointPreflightError("Mamba foundation donor does not match manifest")
        if size != 128256:
            raise CheckpointPreflightError("Mamba lexical donor requires 128256 token IDs")
        required_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        label = "mamba3-mimo-1.5b / Llama-3.1 token IDs"
    elif method == "direct_cross_architecture_weight_transplant":
        donors = manifest.get("donors", {})
        language = donors.get("language", {}) if isinstance(donors, dict) else {}
        if not isinstance(language, dict) or language.get("repo") != "openai/gpt-oss-20b":
            raise CheckpointPreflightError("GPT-OSS donor identity mismatches lexical manifest")
        if size != 201088:
            raise CheckpointPreflightError("GPT-OSS lexical donor requires 201088 token IDs")
        # Exact IDs are set by the GPT-OSS tokenizer asset; do not impose
        # Llama special IDs on the GPT-OSS vocabulary.
        required_tokens = {
            "<|im_start|>": None,
            "<|im_end|>": None,
        }
        label = "gpt-oss-20b lexical IDs"
    else:
        raise CheckpointPreflightError(
            f"unsupported lexical source method {method!r}; add a verified source mapping before serving"
        )
    if not callable(getattr(tokenizer, "get_vocab", None)):
        raise CheckpointPreflightError("tokenizer lacks complete token/ID vocabulary")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise CheckpointPreflightError("tokenizer mapping is empty/invalid")
    if len(vocab) != size or len(set(vocab.values())) != size:
        raise CheckpointPreflightError("tokenizer has duplicate or missing token IDs")
    if any(type(i) is not int or i < 0 or i >= size for i in vocab.values()):
        raise CheckpointPreflightError("tokenizer mapping has out-of-range token IDs")
    for token, expected_id in required_tokens.items():
        actual = vocab.get(token)
        if actual is None or (expected_id is not None and actual != expected_id):
            raise CheckpointPreflightError(f"tokenizer special ID mismatch for {token}")
    return label
