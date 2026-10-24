"""Real GLM-5.3 -> IQ byte-aligned activation artifacts for MOHAWK Stage 2.

Requires measured teacher hidden states and an independently held-out-validated
residual_out coordinate map. No synthetic activations, future-token leakage,
or direct substitution of GLM token IDs into the Mamba/IQ vocabulary.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .alignment import TokenByteSpan, align_glm_states_to_mamba_tokens
from .transport import CoordinateMap, save_coordinate_map


class Stage2ArtifactError(ValueError):
    pass


def write_stage2_example(
    output_dir: str | Path,
    *,
    document_bytes: bytes,
    glm_states: np.ndarray,
    glm_token_spans: list[tuple[int, int]],
    iq_token_spans: list[tuple[int, int]],
    iq_token_ids: list[int],
    residual_out_map: CoordinateMap,
    source_glm_layer: int,
    target_iq_layer: int,
    teacher_revision: str,
    split: str,
) -> dict:
    if split not in ("train", "validation"):
        raise Stage2ArtifactError("split must be train or validation")
    if not document_bytes:
        raise Stage2ArtifactError("document must be nonempty")
    try:
        document_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage2ArtifactError("document must be valid UTF-8") from exc
    if source_glm_layer < 0 or target_iq_layer < 0:
        raise Stage2ArtifactError("source/target layers must be nonnegative")
    if len(teacher_revision) != 40 or any(c not in "0123456789abcdef" for c in teacher_revision.lower()):
        raise Stage2ArtifactError("pinned 40-character GLM revision is required")
    if not iq_token_ids or len(iq_token_ids) != len(iq_token_spans):
        raise Stage2ArtifactError("IQ IDs and byte spans must match exactly")
    if not all(isinstance(x, int) and 0 <= x < 128256 for x in iq_token_ids):
        raise Stage2ArtifactError("IQ IDs must be 0..128255")
    source_space = f"glm53.layer.{source_glm_layer}.residual_out"
    target_space = f"iq.layer.{target_iq_layer}.residual_out"
    if (
        residual_out_map.source_space != source_space
        or residual_out_map.target_space != target_space
    ):
        raise Stage2ArtifactError(
            f"expected measured map {source_space} -> {target_space}"
        )
    raw = np.asarray(glm_states)
    if raw.ndim != 2 or raw.shape[0] != len(glm_token_spans):
        raise Stage2ArtifactError("GLM states [GLM tokens, hidden] do not match spans")
    if residual_out_map.matrix.shape[1] != 4096:
        raise Stage2ArtifactError("coordinate map must project into IQ's 4096-D frame")
    digest = sha256(document_bytes).hexdigest()
    source = [
        TokenByteSpan(token_index=i, start=int(begin), end=int(end))
        for i, (begin, end) in enumerate(glm_token_spans)
    ]
    target = [
        TokenByteSpan(token_index=i, start=int(begin), end=int(end))
        for i, (begin, end) in enumerate(iq_token_spans)
    ]
    aligned = align_glm_states_to_mamba_tokens(
        raw,
        source,
        target,
        source_document_sha256=digest,
        target_document_sha256=digest,
        coordinate_map=residual_out_map,
    )
    if max(s.end for s in source) != len(document_bytes):
        raise Stage2ArtifactError("GLM token spans do not cover document bytes")
    if max(s.end for s in target) != len(document_bytes):
        raise Stage2ArtifactError("IQ token spans do not cover document bytes")

    path = Path(output_dir)
    if path.exists():
        raise Stage2ArtifactError("output exists; refusing to overwrite alignment")
    path.mkdir(parents=True)
    target_hidden = torch.from_numpy(aligned.teacher_aligned_hidden).float()
    mask = torch.from_numpy(aligned.valid_token_mask).bool()
    tensors = {
        "input_ids": torch.tensor(iq_token_ids, dtype=torch.long).unsqueeze(0),
        "teacher_aligned_hidden": target_hidden.unsqueeze(0).contiguous(),
        "teacher_alignment_mask": mask.unsqueeze(0).contiguous(),
    }
    artifact = path / "alignment.safetensors"
    save_file(tensors, str(artifact))
    map_path = path / "validated-residual-map"
    map_tensor, map_json = save_coordinate_map(residual_out_map, map_path)
    def sha(path: Path) -> str:
        h = sha256()
        with path.open("rb") as f:
            for part in iter(lambda: f.read(1024 * 1024), b""):
                h.update(part)
        return h.hexdigest()
    manifest = {
        "schema_version": 1,
        "stage": "glm53_mohawk_stage2",
        "split": split,
        "document_sha256": digest,
        "source_glm_layer": source_glm_layer,
        "target_iq_layer": target_iq_layer,
        "teacher_revision_claim": teacher_revision,
        "teacher_capture_authenticated": False,
        "alignment_tokens": len(iq_token_ids),
        "causally_aligned_tokens": int(mask.sum().item()),
        "alignment_sha256": sha(artifact),
        "map_sha256": sha(map_tensor),
        "map_metadata_sha256": sha(map_json),
        "heldout_validation_rmse": residual_out_map.diagnostics.validation_rmse,
        "quality_verified": False,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
