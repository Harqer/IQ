from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .checkpoint import SafetensorsSource
from .glm53 import GLM53Inspector
from .job import hash_tokenizer_files
from .manifest import DonorManifest, build_donor_manifest


class GLM53TransferError(RuntimeError):
    pass


@dataclass(frozen=True)
class GLM53DonorArtifact:
    manifest: DonorManifest
    checkpoint_dir: Path


def validate_glm53_donor(
    *,
    checkpoint: str | Path,
    checkpoint_revision: str,
    donor_license: str,
    source_uri: str | None = None,
    require_bf16: bool = True,
    manifest_output: str | Path | None = None,
) -> GLM53DonorArtifact:
    root = Path(checkpoint)
    config_path = root / "config.json"
    try:
        config_data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GLM53TransferError(
            f"GLM-5.3 config.json is missing or invalid: {config_path}"
        ) from exc
    if not isinstance(config_data, dict):
        raise GLM53TransferError("GLM-5.3 config.json must contain an object")

    if require_bf16 and config_data.get("quantization_config") is not None:
        raise GLM53TransferError(
            "canonical GLM-5.3 weight transport requires the BF16 donor; "
            "the supplied checkpoint declares quantization_config"
        )

    inspector = GLM53Inspector.from_config_mapping(config_data)
    if require_bf16 and str(inspector.config.dtype).lower() not in {
        "bfloat16",
        "bf16",
        "torch.bfloat16",
    }:
        raise GLM53TransferError(
            f"canonical GLM-5.3 donor must be BF16, got dtype={inspector.config.dtype!r}"
        )

    source = SafetensorsSource(root)
    report = inspector.validate_checkpoint(source)
    report.require_ok()
    if require_bf16 and report.warnings:
        raise GLM53TransferError("; ".join(report.warnings))

    tokenizer_hash = hash_tokenizer_files(root)
    manifest = build_donor_manifest(
        inspector.config,
        source,
        donor_id="glm53",
        checkpoint_revision=checkpoint_revision,
        tokenizer_hash=tokenizer_hash,
        license=donor_license,
        operator_layout_version="glm53-mla-dsa-moe-v1",
        source_uri=source_uri or root.resolve().as_uri(),
    )
    if manifest_output is not None:
        manifest.write_json(manifest_output)
    return GLM53DonorArtifact(
        manifest=manifest,
        checkpoint_dir=root,
    )
