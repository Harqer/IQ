"""Bind full78 incremental shards to their exact original calibration and donors."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class GLM53ProvenanceError(RuntimeError):
    pass


FILE = "run-provenance.json"


def expected_provenance(calibration_dir: Path, *, glm_revision: str,
                        mamba_revision: str, target_fingerprint: str) -> dict:
    receipt = calibration_dir / "calibration_sha256.json"
    if not receipt.is_file():
        raise GLM53ProvenanceError("missing complete calibration SHA256 receipt")
    try:
        hashes = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GLM53ProvenanceError("malformed calibration SHA256 receipt") from exc
    if hashes.get("schema_version") != 1 or len(hashes.get("files", {})) != 866:
        raise GLM53ProvenanceError("calibration receipt must cover exactly 866 files")
    return {
        "schema_version": 1,
        "calibration_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "teacher_revision": glm_revision,
        "mamba_revision": mamba_revision,
        "target_fingerprint": target_fingerprint,
    }


def enforce_run_provenance(output: Path, calibration: Path, *,
                           glm_revision: str, mamba_revision: str,
                           target_fingerprint: str, resume: bool) -> dict:
    expected = expected_provenance(
        calibration, glm_revision=glm_revision,
        mamba_revision=mamba_revision, target_fingerprint=target_fingerprint
    )
    path = output / FILE
    if resume:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GLM53ProvenanceError(
                "unsafe legacy checkpoint: no verified run provenance; use a NEW run"
            ) from exc
        if previous != expected:
            raise GLM53ProvenanceError(
                "resume provenance mismatch: original calibration or donor changed"
            )
        return expected

    if path.exists() or path.is_symlink():
        raise GLM53ProvenanceError("refuse to overwrite existing run provenance")
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".run-provenance.", suffix=".partial",
                                    dir=str(output))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(expected, out, sort_keys=True, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return expected
