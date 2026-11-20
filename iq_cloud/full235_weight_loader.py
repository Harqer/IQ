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
import os
from pathlib import Path
import re
from typing import Callable, Mapping
from .inference_diagnostics import span_scope, emit_progress

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
    verified_remote_shard, snapshot_remote_shard,
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
    loaded_lexical_tensor_count: int = 0
    sampled_lexical_rows_verified: int = 0


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
                    != expected_lexical_shape[0]
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
    install_source_lexical: bool = False,
) -> set[str]:
    """Copy nonlexical tensors, optionally original-ID lexical tensors (source mode only)."""
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
            if name in SOURCE_LEXICAL and not install_source_lexical:
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


@torch.no_grad()
def verify_source_lexical_relation(
    targets: Mapping[str, torch.Tensor],
    *,
    rows: tuple[int, ...] = (0, 1, 100, 1000, 64000, 128000, 128255),
) -> int:
    """Sample the two widened halves and 2x output-head scaling in original IDs.

    This is structural evidence, not proof of equality to the original donor.
    """
    embedding = targets["embed_tokens.weight"]
    head = targets["lm_head.weight"]
    if (embedding.shape != head.shape or embedding.ndim != 2
            or embedding.shape[1] % 2):
        raise VerifiedBodyLoaderError("source lexical tensor geometry mismatch")
    half = embedding.shape[1] // 2
    for row in rows:
        if type(row) is not int or not 0 <= row < embedding.shape[0]:
            raise VerifiedBodyLoaderError("invalid lexical verification row")
        emb, lm = embedding[row], head[row]
        if (not bool(torch.isfinite(emb).all().item())
                or not bool(torch.isfinite(lm).all().item())
                or not bool((emb != 0).any().item())
                or not torch.equal(emb[:half], emb[half:])
                or not torch.equal(lm[:half], lm[half:])
                or not torch.equal(emb, lm * 2)):
            raise VerifiedBodyLoaderError(
                f"source lexical widening/scaling mismatch at row {row}"
            )
    return len(rows)


def _read_bounded(fs, name: str, limit: int) -> bytes:
    with fs.open(f"{ROOT}/{name}", "rb") as f:
        raw = f.read(limit+1)
    if len(raw) > limit:
        raise VerifiedBodyLoaderError("remote checkpoint metadata exceeds allowed size")
    return raw


def write_unreviewed_snapshot(
    path: str | Path, *, hashes: Mapping[str, str],
    indexed_shards: set[str], source_index_digest: str = EXPECTED_INDEX_SHA256,
) -> str:
    """Write a complete experimental manifest once; never label it trusted.

    The caller must persist it outside an ephemeral worker for future review.
    An incomplete manifest is not saved and an existing file is never replaced.
    """
    manifest = {
        "schema_version": 1,
        "source_index_sha256": source_index_digest,
        "sha256_algorithm": "sha256",
        "model_source_status": "compiled_not_inference_validated",
        "shards_sha256": dict(hashes),
        "status": "unreviewed_source_digest_snapshot",
        "full_tensor_finiteness_checked": True,
        "original_llama_lexical_structure_sampled": True,
        "independently_pinned": False,
        "inference_ready": False,
    }
    validate_digest_manifest(
        manifest, indexed_shards, source_index_digest=source_index_digest,
    )
    path = Path(path)
    if path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise VerifiedBodyLoaderError(
            "unreviewed manifest destination must be an unused file in an existing directory"
        )
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return sha256(data).hexdigest()


def load_verified_full235_body(
    model: torch.nn.Module,
    *,
    approved_manifest_bytes: bytes | None = None,
    approved_manifest_sha256: str | None = None,
    scratch_dir: str | None = None,
    on_shard_verified: Callable[[int, int], None] | None = None,
    install_source_lexical: bool = False,
    experimental_snapshot_path: str | None = None,
    tracer=None,
) -> LoadedSourceBody:
    """Single-pass shard hashing/injection: trusted or explicitly experimental.

    EXPERIMENTAL: captures an unreviewed SHA-256 map while streaming original
    source tensors once; never claims independent provenance, even on success.
    PRODUCTION: requires an independently pinned digest manifest before reads.
    """
    experimental = experimental_snapshot_path is not None
    if experimental:
        if (not install_source_lexical
                or approved_manifest_bytes is not None
                or approved_manifest_sha256 is not None
                or os.getenv("IQ235_EXPLORATORY_UNPINNED") != "I_ACCEPT_UNATTESTED_SOURCE"
                or os.getenv("IQ235_AUTHORIZE_539GB_READ") != "I_AUTHORIZE_539GB"):
            raise VerifiedBodyLoaderError(
                "single-pass exploratory source loading requires separate explicit "
                "acceptance and 539GB read authorization"
            )
        target = Path(experimental_snapshot_path)
        if (not target.is_absolute() or target.is_symlink()
                or target.exists() or not target.parent.is_dir()):
            raise VerifiedBodyLoaderError(
                "unreviewed manifest destination must be unused absolute path"
            )
        manifest = None
    else:
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

    glm_config = read_architecture_contract()
    original_config = json.loads(config_paths()[0].read_text(encoding="utf-8"))
    target_config = original_config if install_source_lexical else glm_config
    fs = HfFileSystem()
    config_bytes = _read_bounded(fs, "iq_config.json", 100_000)
    index_bytes = _read_bounded(fs, "model.safetensors.index.json", 8_000_000)
    marker_bytes = _read_bounded(fs, "COMPILE_COMPLETE.json", 100_000)
    check_pinned_metadata_sha256(config_bytes, index_bytes)
    if json.loads(config_bytes) != original_config:
        raise VerifiedBodyLoaderError("remote source config differs from pinned checkpoint")
    index = json.loads(index_bytes)
    build_migration_plan(original_config, glm_config, index,
                         json.loads(marker_bytes))
    indexed_shards = set(index["weight_map"].values())
    hashes = (
        {} if experimental else validate_digest_manifest(
            manifest, indexed_shards, source_index_digest=EXPECTED_INDEX_SHA256,
        )
    )
    # Fail before any 539 GB network streaming if model geometry is wrong.
    targets = validate_model_inventory(
        model, index, expected_config=target_config,
        expected_lexical_shape=(SOURCE_LEXICAL_SHAPE if install_source_lexical
                                else TARGET_LEXICAL_SHAPE),
    )
    grouped: dict[str, set[str]] = defaultdict(set)
    for key, shard in index["weight_map"].items():
        grouped[shard].add(key)
    if len(grouped) != EXPECTED_SHARDS or len(targets) != EXPECTED_KEYS:
        raise VerifiedBodyLoaderError("source shard/tensor coverage inconsistent")
    written: set[str] = set()
    # Source model may be partially loaded if an I/O error occurs. No attestation
    # is ever emitted on failure, and the caller MUST discard the model object.
    with span_scope(tracer, "iq235.checkpoint.stream_verified_shards", {
        "iq.shards.expected": EXPECTED_SHARDS,
        "iq.source_lexical": install_source_lexical,
        "iq.integrity.mode": "unreviewed_snapshot" if experimental else "pinned",
    }) as checkpoint_span:
        for n, (name, keys) in enumerate(sorted(grouped.items()), 1):
            if experimental:
                with snapshot_remote_shard(
                    fs, f"{ROOT}/{name}", expected_keys=keys,
                    scratch_dir=scratch_dir,
                ) as (staged, header, actual_sha256):
                    installed = install_verified_shard(
                        staged, header, keys, targets,
                        install_source_lexical=True,
                    )
                hashes[name] = actual_sha256
            else:
                with verified_remote_shard(
                    fs, f"{ROOT}/{name}", hashes[name],
                    expected_keys=keys, scratch_dir=scratch_dir,
                ) as (staged, header):
                    installed = install_verified_shard(
                        staged, header, keys, targets,
                        install_source_lexical=install_source_lexical,
                    )
            if written & installed:
                raise VerifiedBodyLoaderError("source tensor installed more than once")
            written |= installed
            emit_progress(checkpoint_span, n, EXPECTED_SHARDS, len(written))
            if on_shard_verified is not None:
                on_shard_verified(n, EXPECTED_SHARDS)
        expected = set(targets) if install_source_lexical else set(targets) - SOURCE_LEXICAL
        if written != expected:
            raise VerifiedBodyLoaderError("full235 source checkpoint tensor coverage incomplete")
        sampled_rows = (
            verify_source_lexical_relation(targets)
            if install_source_lexical else 0
        )
        manifest_sha256 = (
            write_unreviewed_snapshot(
                experimental_snapshot_path, hashes=hashes,
                indexed_shards=indexed_shards,
            ) if experimental else approved_manifest_sha256
        )
        if checkpoint_span is not None:
            checkpoint_span.set_attribute("iq.shards.completed", len(grouped))
            checkpoint_span.set_attribute("iq.tensors.loaded", len(written))
            checkpoint_span.set_attribute("iq.lexical.sampled_rows", sampled_rows)
            checkpoint_span.set_attribute(
                "iq.integrity.independently_verified", not experimental,
            )
    return LoadedSourceBody(
        model=model,
        loaded_nonlexical_tensor_count=len(written - SOURCE_LEXICAL),
        verified_source_shard_count=len(grouped),
        source_index_sha256=EXPECTED_INDEX_SHA256,
        digest_manifest_sha256=manifest_sha256,
        status=("unreviewed_full_source_snapshot_loaded_quality_unvalidated"
                if experimental else
                "verified_full_source_checkpoint_loaded_quality_unvalidated"
                if install_source_lexical else
                "verified_source_body_loaded_glm_lexical_untrained"),
        lexical_rows_initialized=install_source_lexical,
        loaded_lexical_tensor_count=(2 if install_source_lexical else 0),
        sampled_lexical_rows_verified=sampled_rows,
    )
