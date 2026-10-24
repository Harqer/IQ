"""Reuse immutable IQ Mamba-3 and lexical shards in GLM expansion compiles.

The existing 32 Mamba shards and global embedding/head are copied byte-for-byte,
not re-transplanted from the original raw 1.5B donor. This is deliberately
independent from the GLM donor and its expanded expert count.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

from iq_model import IQHybridConfig


class FoundationReuseError(ValueError):
    pass


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reuse_frozen_mamba_foundation(
    *,
    original: Path,
    output: Path,
    expected_source_fingerprint: str,
    target_config: IQHybridConfig,
    mamba_revision: str,
    glm_revision: str,
    weight_map: dict[str, str],
) -> dict[str, object]:
    from .complete_transplant import canonical_glm53_config

    root = original.resolve(strict=True)
    dest = output.resolve(strict=True)
    if root == dest or root in dest.parents or dest in root.parents:
        raise FoundationReuseError("candidate destination must not overlap original checkpoint")
    old = IQHybridConfig.from_json(str(root / "iq_config.json"))
    if old.fingerprint != expected_source_fingerprint:
        raise FoundationReuseError("original IQ config does not match known frozen source")
    if old.fingerprint != canonical_glm53_config().fingerprint:
        raise FoundationReuseError("original IQ does not use canonical 32-expert Mamba foundation")
    if (
        old.mamba3 != target_config.mamba3
        or old.schedule != target_config.schedule
        or old.model != target_config.model
        or old.compressed_context != target_config.compressed_context
        or old.attnres != target_config.attnres
    ):
        raise FoundationReuseError("expanded candidate changes frozen Mamba/lexical/context geometry")
    provenance = json.loads((root / "transplant_manifest.json").read_text(encoding="utf-8"))
    if (
        provenance.get("foundation", {}).get("repo") != "state-spaces/mamba3-mimo-1.5b"
        or provenance["foundation"].get("revision") != mamba_revision
        or provenance.get("capability_source", {}).get("revision") != glm_revision
    ):
        raise FoundationReuseError("original pretrained donor revisions differ from candidate")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    original_map = index.get("weight_map")
    if not isinstance(original_map, dict) or not original_map:
        raise FoundationReuseError("original IQ index is missing weight_map")
    files = ["model-global.safetensors"] + [
        f"model-mamba-{i:03d}.safetensors" for i in range(32)
    ]
    expected = set(files)
    keys_by_file: dict[str, list[str]] = {f: [] for f in files}
    for name, file in original_map.items():
        if file in expected:
            keys_by_file[file].append(name)
    if set(keys_by_file["model-global.safetensors"]) != {
        "embed_tokens.weight", "norm.weight", "lm_head.weight"
    }:
        raise FoundationReuseError("original lexical foundation index keys are incorrect")
    mamba_positions = target_config.schedule.positions(
        __import__("iq_model", fromlist=["HybridLayerType"]).HybridLayerType.MAMBA3
    )
    if len(mamba_positions) != 32:
        raise FoundationReuseError("recipient Mamba schedule changed")
    for ordinal, physical in enumerate(mamba_positions):
        filename = f"model-mamba-{ordinal:03d}.safetensors"
        keys = keys_by_file[filename]
        prefix = f"layers.{physical}."
        if not keys or not all(
            key.startswith(prefix) and (
                key == f"{prefix}norm.weight" or key.startswith(f"{prefix}mamba.core.")
            )
            for key in keys
        ):
            raise FoundationReuseError(f"missing/mismatched original Mamba layer keys: {filename}")
    file_digests: dict[str, str] = {}
    for filename in files:
        src = root / filename
        dst = dest / filename
        if not src.is_file() or src.is_symlink():
            raise FoundationReuseError(f"original Mamba shard missing or symlinked: {filename}")
        if dst.exists():
            raise FoundationReuseError(f"candidate foundation shard already exists: {filename}")
        for key in keys_by_file[filename]:
            if key in weight_map:
                raise FoundationReuseError(f"duplicate target tensor during Mamba reuse: {key}")
        src_digest = _hash_file(src)
        shutil.copyfile(src, dst)
        if _hash_file(dst) != src_digest:
            raise FoundationReuseError(f"copy verification failed: {filename}")
        file_digests[filename] = src_digest
        for key in keys_by_file[filename]:
            weight_map[key] = filename
    return {
        "method": "bitwise_reuse_immutable_iq_mamba_foundation",
        "parent_checkpoint_index_sha256": _hash_file(root / "model.safetensors.index.json"),
        "file_sha256": file_digests,
        "foundation_shards_reused": 33,
        "mamba_pretrained_weights_unchanged": True,
        "full_hybrid_behavior_preserved": False,
    }
