"""Verified remote full235 BODY loader; never an inference-ready GLM overlay.

Run only on an authorized remote H200x8 worker. The 539 GB source stays in
the Hugging Face bucket; each shard is temporarily streamed, SHA-256 verified,
copied to its exact existing model tensor on its assigned GPU, then deleted.

No trusted per-shard digest manifest currently exists in the source bucket.
Consequently the live loader MUST fail closed until an independently reviewed
digest manifest is pinned via the worker's trusted deployment configuration.
No metadata-only assertion or caller-provided boolean can enable inference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Callable, Mapping

import torch

from .checkpoint_metadata import (
    BUCKET, RUN, EXPECTED_CONFIG_SHA256, EXPECTED_INDEX_SHA256,
    EXPECTED_KEYS, EXPECTED_SHARDS, IQ235MetadataError,
    check_pinned_metadata_sha256,
    PINNED_SOURCE_SHARD_MANIFEST_SHA256,
)
from .checkpoint_migration import build_migration_plan
from .glm53_lexical import GLM_VOCAB, IQ_WIDTH, SOURCE_IQ_VOCAB
from .remote_shard_loader import (
    RemoteShardError, SHA256, validate_digest_manifest,
    verified_remote_shard,
)

# The payload checksum pin is owned by the pure-metadata module so GPU-less
# readiness checks never need to import torch merely to read this value.
SOURCE_LEXICAL = frozenset(("embed_tokens.weight", "lm_head.weight"))
SOURCE_LEXICAL_SHAPE = (SOURCE_IQ_VOCAB, IQ_WIDTH)
TARGET_LEXICAL_SHAPE = (GLM_VOCAB, IQ_WIDTH)
TORCH_TO_SAFE = {
    torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8",
    torch.int16: "I16", torch.int32: "I32", torch.int64: "I64",
    torch.float16: "F16", torch.bfloat16: "BF16",
    torch.float32: "F32", torch.float64: "F64",
}
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
ROOT = f"hf://buckets/{BUCKET}/{RUN}"


class VerifiedBodyLoaderError(ValueError):
    """Source tensor cannot safely be copied into target or attested."""


@dataclass(frozen=True)
class LoadedSourceBody:
    """Return ONLY after every nonlexical source tensor has been copied.

    This does not certify the GLM lexical rows or authorize inference.
    """
    model: torch.nn.Module
    loaded_nonlexical_tensor_count: int
    verified_source_shard_count: int
    source_index_sha256: str
    digest_manifest_sha256: str
    status: str = "verified_source_body_loaded_glm_lexical_untrained"
    lexical_rows_initialized: bool = False
    inference_ready: bool = False


def validate_model_inventory(
    model: torch.nn.Module,
    index: Mapping[str, object],
    *,
    expected_lexical_shape: tuple[int, int] = TARGET_LEXICAL_SHAPE,
    expected_config: Mapping[str, object] | None = None,
    require_production: bool = True,
) -> dict[str, torch.Tensor]:
    """Match the source index to live destination parameters exactly."""
    if not isinstance(model, torch.nn.Module):
        raise VerifiedBodyLoaderError("expected a real Torch module")
    weights = model.state_dict()
    source_map = index.get("weight_map")
    if not isinstance(source_map, dict) or set(weights) != set(source_map):
        raise VerifiedBodyLoaderError("live model/index tensor keysets disagree")
    if any(name not in weights for name in SOURCE_LEXICAL):
        raise VerifiedBodyLoaderError("target lexical parameters missing")
    if tuple(weights["embed_tokens.weight"].shape) != expected_lexical_shape:
        raise VerifiedBodyLoaderError("target GLM embedding geometry differs")
    if tuple(weights["lm_head.weight"].shape) != expected_lexical_shape:
        raise VerifiedBodyLoaderError("target GLM output head geometry differs")
    if (weights["embed_tokens.weight"].data_ptr()
            == weights["lm_head.weight"].data_ptr()):
        raise VerifiedBodyLoaderError("GLM lexical matrices must remain untied")
    if require_production:
        model_config = getattr(model, "config", None)
        if (model_config is None
                or not hasattr(model_config, "to_dict")
                or expected_config is None
                or model_config.to_dict() != expected_config
                or getattr(getattr(model_config, "model", None), "vocab_size", None)
                    != GLM_VOCAB
                or getattr(getattr(model_config, "model", None), "hidden_size", None)
                    != IQ_WIDTH
                or len(getattr(model, "layers", ())) != 235
                or len(getattr(getattr(model_config, "schedule", None), "layers", ())) != 235):
            raise VerifiedBodyLoaderError("full235 production architecture not constructed")
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 8
                or any("H200" not in torch.cuda.get_device_name(i)
                       or torch.cuda.get_device_capability(i) != (9, 0)
                       for i in range(8))):
            raise VerifiedBodyLoaderError("exactly eight H200 worker GPUs required")
        for name, tensor in weights.items():
            match = _LAYER_RE.match(name)
            expected_gpu = int(match.group(1)) % 8 if match else 0
            if (tensor.device.type != "cuda"
                    or tensor.device.index != expected_gpu):
                raise VerifiedBodyLoaderError(
                    f"source tensor device placement mismatch: {name}"
                )
    return weights


@torch.no_grad()
def install_verified_shard(
    staged_path: Path,
    header: Mapping[str, Mapping],
    expected_keys: set[str],
    targets: Mapping[str, torch.Tensor],
    *,
    original_lexical_shape: tuple[int, int] = SOURCE_LEXICAL_SHAPE,
    require_original_bf16: bool = True,
) -> set[str]:
    """Copy only real nonlexical source rows; NEVER install source lexical IDs."""
    from safetensors import safe_open

    if set(header) != expected_keys:
        raise VerifiedBodyLoaderError("verified header/index keyset disagreement")
    if any(name not in targets for name in expected_keys):
        raise VerifiedBodyLoaderError("verified source tensor absent from live IQ model")
    # Preflight shapes and dtypes BEFORE copying anything from this shard.
    for name in expected_keys:
        entry = header[name]
        if name in SOURCE_LEXICAL:
            if (tuple(entry["shape"]) != original_lexical_shape
                    or (require_original_bf16 and entry["dtype"] != "BF16")):
                raise VerifiedBodyLoaderError("source lexical header geometry/dtype mismatch")
        else:
            target = targets[name]
            dtype = TORCH_TO_SAFE.get(target.dtype)
            if (tuple(target.shape) != tuple(entry["shape"])
                    or dtype != entry["dtype"]):
                raise VerifiedBodyLoaderError(
                    f"source/target tensor shape or dtype mismatch: {name}"
                )
    written: set[str] = set()
    with safe_open(str(staged_path), framework="pt", device="cpu") as archive:
        if set(archive.keys()) != expected_keys:
            raise VerifiedBodyLoaderError("decoded shard tensor names mismatch")
        for name in sorted(expected_keys):
            if name in SOURCE_LEXICAL:
                continue
            value = archive.get_tensor(name)
            target = targets[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise VerifiedBodyLoaderError(f"decoded source dtype/shape mismatch: {name}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise VerifiedBodyLoaderError(f"source tensor has nonfinite values: {name}")
            target.copy_(value.to(device=target.device))
            written.add(name)
    return written


def _read_bounded(fs, name: str, limit: int) -> bytes:
    with fs.open(f"{ROOT}/{name}", "rb") as f:
        raw = f.read(limit+1)
    if len(raw) > limit:
        raise VerifiedBodyLoaderError("remote checkpoint metadata exceeds allowed size")
    return raw


def load_verified_full235_body(
    model: torch.nn.Module,
    *,
    approved_manifest_bytes: bytes,
    approved_manifest_sha256: str,
    scratch_dir: str | None = None,
    on_shard_verified: Callable[[int, int], None] | None = None,
) -> LoadedSourceBody:
    """Production-only source injection; refuses unpinned/missing digest files.

    Never call this with a randomly constructed production model expecting it
    to generate meaningful GLM text: source lexical matrix IDs differ and
    final full235 inference is still forbidden until lexical training/validation.
    """
    if (PINNED_SOURCE_SHARD_MANIFEST_SHA256 is None
            or approved_manifest_sha256 != PINNED_SOURCE_SHARD_MANIFEST_SHA256):
        raise VerifiedBodyLoaderError(
            "no independently pinned 20,283-shard manifest; production loading blocked"
        )
    if (not isinstance(approved_manifest_bytes, bytes)
            or not isinstance(approved_manifest_sha256, str)
            or SHA256.fullmatch(approved_manifest_sha256) is None
            or sha256(approved_manifest_bytes).hexdigest() != approved_manifest_sha256):
        raise VerifiedBodyLoaderError("independently approved shard digest manifest required")
    try:
        manifest = json.loads(approved_manifest_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerifiedBodyLoaderError("unreadable trusted digest manifest") from exc

    from huggingface_hub import HfFileSystem
    from iq_cloud.full_iq235 import read_architecture_contract, config_paths

    target_config = read_architecture_contract()
    original_config = json.loads(config_paths()[0].read_text(encoding="utf-8"))
    fs = HfFileSystem()
    config_bytes = _read_bounded(fs, "iq_config.json", 100_000)
    index_bytes = _read_bounded(fs, "model.safetensors.index.json", 8_000_000)
    marker_bytes = _read_bounded(fs, "COMPILE_COMPLETE.json", 100_000)
    check_pinned_metadata_sha256(config_bytes, index_bytes)
    if json.loads(config_bytes) != original_config:
        raise VerifiedBodyLoaderError("remote source config differs from pinned checkpoint")
    index = json.loads(index_bytes)
    build_migration_plan(original_config, target_config, index,
                         json.loads(marker_bytes))
    hashes = validate_digest_manifest(
        manifest, set(index["weight_map"].values()),
        source_index_digest=EXPECTED_INDEX_SHA256,
    )
    # Fail before any 539 GB network streaming if model geometry is wrong.
    targets = validate_model_inventory(
        model, index, expected_config=target_config,
    )
    grouped: dict[str, set[str]] = defaultdict(set)
    for key, shard in index["weight_map"].items():
        grouped[shard].add(key)
    if len(grouped) != EXPECTED_SHARDS or len(targets) != EXPECTED_KEYS:
        raise VerifiedBodyLoaderError("source shard/tensor coverage inconsistent")
    written: set[str] = set()
    # Source model may be partially loaded if an I/O error occurs. No attestation
    # is ever emitted on failure, and the caller MUST discard the model object.
    for n, (name, keys) in enumerate(sorted(grouped.items()), 1):
        with verified_remote_shard(
            fs, f"{ROOT}/{name}", hashes[name],
            expected_keys=keys, scratch_dir=scratch_dir,
        ) as (staged, header):
            installed = install_verified_shard(staged, header, keys, targets)
        if written & installed:
            raise VerifiedBodyLoaderError("source tensor installed more than once")
        written |= installed
        if on_shard_verified is not None:
            on_shard_verified(n, EXPECTED_SHARDS)
    if written != set(targets) - SOURCE_LEXICAL:
        raise VerifiedBodyLoaderError("full235 pretrained source body not completely installed")
    return LoadedSourceBody(
        model=model,
        loaded_nonlexical_tensor_count=len(written),
        verified_source_shard_count=len(grouped),
        source_index_sha256=EXPECTED_INDEX_SHA256,
        digest_manifest_sha256=approved_manifest_sha256,
    )
