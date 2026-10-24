"""Strictly apply a separately trained, unpromoted IQ parameter delta in memory.

Never edits the immutable checkpoint; only explicitly listed and shape-checked
model parameters are changed. Runtime still must pass real generation QA.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


class CandidateDeltaError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def apply_candidate_delta(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    delta_dir: str | Path,
) -> int:
    base = Path(checkpoint_dir)
    overlay = Path(delta_dir)
    manifest = json.loads((overlay / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise CandidateDeltaError("unsupported delta manifest schema")
    if manifest.get("status") != "candidate_not_promoted" or manifest.get("language_quality_verified") is not False:
        raise CandidateDeltaError("only explicitly unpromoted/untested candidate deltas are supported")
    index_hash = _sha(base / "model.safetensors.index.json")
    pinned_hash = manifest.get("base_index_sha256", manifest.get("parent_checkpoint_index_sha256"))
    if index_hash != pinned_hash:
        raise CandidateDeltaError("delta belongs to a different immutable IQ checkpoint")
    stage = manifest.get("target_iq_layer")
    if stage is not None:
        allowed_prefixes = (f"layers.{stage-1}.", f"layers.{stage}.")
        delta_file = overlay / "stage2_delta.safetensors"
    elif "trainable_layers" in manifest:
        # Behavioral run explicitly unfreezes the final GLM context/MoE pair + norm.
        selected = manifest["trainable_layers"]
        if not isinstance(selected, list) or len(selected) != 2:
            raise CandidateDeltaError("invalid behavioral training stage list")
        allowed_prefixes = tuple(f"layers.{int(i)}." for i in selected) + ("norm.",)
        delta_file = overlay / "behavior_delta.safetensors"
    else:
        raise CandidateDeltaError("missing selection of changed parameters")
    if _sha(delta_file) != manifest.get("delta_sha256"):
        raise CandidateDeltaError("delta payload hash mismatch")
    updated = load_file(str(delta_file), device="cpu")
    if not updated or len(updated) != manifest.get("delta_tensors"):
        raise CandidateDeltaError("delta tensor count is incorrect")
    parameters = dict(model.named_parameters())
    for name, tensor in updated.items():
        if not name.startswith(allowed_prefixes) or name not in parameters:
            raise CandidateDeltaError(f"unexpected or non-trainable delta key: {name}")
        if tuple(tensor.shape) != tuple(parameters[name].shape):
            raise CandidateDeltaError(f"wrong tensor shape: {name}")
        if not bool(torch.isfinite(tensor.float()).all()):
            raise CandidateDeltaError(f"non-finite candidate tensor: {name}")
    with torch.no_grad():
        for name, tensor in updated.items():
            target = parameters[name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return len(updated)
