"""Immutable, adapter-only checkpoint format for a GLM-first Mamba hybrid.

Does not store/rewrite GLM or Mamba donor weights. It only records the new
gates and GLM<->Mamba adapters; provenance is needed to reload the source
models. This artifact is NEVER inference-ready by virtue of being saved.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from iq_model.glm_first_mamba import GLMFirstMambaHybrid


class GLMOverlayCheckpointError(RuntimeError):
    pass


def _hex_digest(value: str, length: int, label: str) -> str:
    if not isinstance(value, str):
        raise GLMOverlayCheckpointError(f"{label} must be a pinned hex fingerprint")
    normalized = value.lower().strip()
    if len(normalized) != length or any(c not in "0123456789abcdef" for c in normalized):
        raise GLMOverlayCheckpointError(f"{label} must be a pinned {length}-character hex fingerprint")
    return normalized


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as inp:
        while chunk := inp.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _overlay_tensors(branch) -> dict[str, torch.Tensor]:
    named = {
        "gate": branch.gate,
        "in_adapter.weight": branch.in_adapter.weight,
        "out_adapter.weight": branch.out_adapter.weight,
    }
    if any(not bool(torch.isfinite(t.detach()).all()) for t in named.values()):
        raise GLMOverlayCheckpointError("overlay contains NaN/Inf")
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in named.items()
    }


def save_glm_first_overlay(
    hybrid: GLMFirstMambaHybrid,
    folder: str | Path,
    *,
    glm_revision: str,
    mamba_revision: str,
    tokenizer_mapping_sha256: str,
    training_steps: int,
) -> dict[str, Any]:
    """Create a NEW checkpoint directory. Refuse to replace existing outputs."""
    glm_revision = _hex_digest(glm_revision, 40, "GLM revision")
    mamba_revision = _hex_digest(mamba_revision, 40, "Mamba revision")
    token_hash = _hex_digest(tokenizer_mapping_sha256, 64, "tokenizer ID map")
    if not isinstance(training_steps, int) or training_steps < 0:
        raise GLMOverlayCheckpointError("training_steps must be a nonnegative integer")
    root = Path(folder)
    if root.exists():
        raise GLMOverlayCheckpointError("checkpoint directory already exists: overwrite refused")
    root.mkdir(parents=True, exist_ok=False)
    shards = []
    for index, branch in enumerate(hybrid.branches):
        relative = f"overlay-{index:03d}.safetensors"
        path = root / relative
        # Individual files, bounded CPU staging rather than one giant 24-block copy.
        save_file(_overlay_tensors(branch), str(path))
        shards.append({
            "block": index, "file": relative,
            "sha256": _hash_file(path), "bytes": path.stat().st_size
        })
    report = {
        "format": "iq_glm_first_mamba_overlay_v1",
        "inference_ready": False,
        "glmsource_revision": glm_revision,
        "mamba_source_revision": mamba_revision,
        "tokenizer_mapping_sha256": token_hash,
        "glm_vocab_size": hybrid.contract.glm_vocab,
        "glm_hidden": hybrid.contract.glm_hidden,
        "mamba_hidden": hybrid.contract.mamba_hidden,
        "mamba_rank": hybrid.contract.mimo_rank,
        "mamba_state": hybrid.contract.state_size,
        "mamba_mlp_hidden": hybrid.contract.mamba_mlp_hidden,
        "glm_positions": list(hybrid.placements),
        "training_steps": training_steps,
        "shards": shards,
        "note": "Adapters and gates only. Load ORIGINAL pinned GLM and Mamba donors separately. "
                "This is NOT an inference-ready full-model checkpoint.",
    }
    manifest = root / "overlay_manifest.json"
    temporary = root / ".overlay_manifest.partial"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return report


def load_glm_first_overlay(
    hybrid: GLMFirstMambaHybrid,
    folder: str | Path,
    *,
    glm_revision: str,
    mamba_revision: str,
    tokenizer_mapping_sha256: str,
) -> dict[str, Any]:
    """Verify ALL overlay files and identities before changing any parameter."""
    expected_glm = _hex_digest(glm_revision, 40, "GLM revision")
    expected_mamba = _hex_digest(mamba_revision, 40, "Mamba revision")
    expected_tok = _hex_digest(tokenizer_mapping_sha256, 64, "tokenizer ID map")
    root = Path(folder)
    try:
        manifest = json.loads((root / "overlay_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GLMOverlayCheckpointError("missing or invalid overlay manifest") from exc
    if manifest.get("format") != "iq_glm_first_mamba_overlay_v1":
        raise GLMOverlayCheckpointError("unrecognized overlay checkpoint format")
    for name, expected in (
        ("glmsource_revision", expected_glm),
        ("mamba_source_revision", expected_mamba),
        ("tokenizer_mapping_sha256", expected_tok),
        ("glm_vocab_size", hybrid.contract.glm_vocab),
        ("glm_hidden", hybrid.contract.glm_hidden),
        ("mamba_hidden", hybrid.contract.mamba_hidden),
        ("mamba_rank", hybrid.contract.mimo_rank),
        ("mamba_state", hybrid.contract.state_size),
        ("mamba_mlp_hidden", hybrid.contract.mamba_mlp_hidden),
        ("glm_positions", list(hybrid.placements)),
    ):
        if manifest.get(name) != expected:
            raise GLMOverlayCheckpointError(f"overlay provenance/shape mismatch: {name}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(hybrid.branches):
        raise GLMOverlayCheckpointError("missing/duplicate Mamba overlay blocks")

    # Preflight the entire manifest and checksum every shard without loading
    # the old pretrained model weights or mutating any adapter.
    for i, item in enumerate(shards):
        if not isinstance(item, dict) or item.get("block") != i:
            raise GLMOverlayCheckpointError(f"overlay block order mismatch: {i}")
        name = f"overlay-{i:03d}.safetensors"
        if item.get("file") != name:
            raise GLMOverlayCheckpointError("unexpected overlay shard path")
        path = root / name
        if not path.is_file() or _hash_file(path) != item.get("sha256"):
            raise GLMOverlayCheckpointError(f"missing or corrupt overlay shard: {name}")
        if path.stat().st_size != item.get("bytes"):
            raise GLMOverlayCheckpointError(f"overlay byte count mismatch: {name}")

    # Validate all tensor shapes/finiteness before applying. Record only the
    # small trainable overlay in memory; do not materialize donor weights.
    loaded: list[dict[str, torch.Tensor]] = []
    for i, branch in enumerate(hybrid.branches):
        path = root / f"overlay-{i:03d}.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            required = _overlay_tensors(branch)
            if set(handle.keys()) != set(required):
                raise GLMOverlayCheckpointError(f"overlay tensor keys differ at {i}")
            state = {key: handle.get_tensor(key) for key in required}
        for key, value in state.items():
            if value.shape != required[key].shape or not bool(torch.isfinite(value).all()):
                raise GLMOverlayCheckpointError(
                    f"overlay tensor shape/finite check failed: {i}/{key}"
                )
        loaded.append(state)
    with torch.no_grad():
        for branch, state in zip(hybrid.branches, loaded):
            branch.gate.copy_(state["gate"].to(branch.gate))
            branch.in_adapter.weight.copy_(
                state["in_adapter.weight"].to(branch.in_adapter.weight)
            )
            branch.out_adapter.weight.copy_(
                state["out_adapter.weight"].to(branch.out_adapter.weight)
            )
    return manifest
