from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from hashlib import sha256
import json

from .checkpoint import SafetensorsSource
from .donor import DonorError
from .glm53 import GLM53Inspector
from .job import hash_tokenizer_files
from .manifest import CheckpointFile, DonorManifest, build_donor_manifest


class GLM53TransferError(RuntimeError):
    pass


@dataclass(frozen=True)
class GLM53DonorArtifact:
    manifest: DonorManifest
    checkpoint_dir: Path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(data: Any) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_glm53_streaming_donor(
    *,
    checkpoint: str | Path,
    checkpoint_revision: str,
    donor_license: str,
    source_uri: str | None = None,
    manifest_output: str | Path | None = None,
) -> GLM53DonorArtifact:
    root = Path(checkpoint)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    try:
        config_data: Any = json.loads(config_path.read_text(encoding="utf-8"))
        index_data: Any = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GLM53TransferError(
            "streaming GLM source requires valid config.json and "
            "model.safetensors.index.json"
        ) from exc
    if not isinstance(config_data, dict) or not isinstance(index_data, dict):
        raise GLM53TransferError("streaming GLM metadata must contain JSON objects")
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise GLM53TransferError("GLM safetensors index is missing weight_map")

    inspector = GLM53Inspector.from_config_mapping(config_data)
    if config_data.get("quantization_config") is not None:
        raise GLM53TransferError(
            "canonical GLM-5.3 streaming transport requires the BF16 source"
        )
    if str(inspector.config.dtype).lower() not in {
        "bfloat16",
        "bf16",
        "torch.bfloat16",
    }:
        raise GLM53TransferError(
            f"canonical GLM-5.3 donor must be BF16, got dtype={inspector.config.dtype!r}"
        )
    report = inspector.validate_index_keys(tuple(str(k) for k in weight_map))
    try:
        report.require_ok()
    except DonorError as exc:
        raise GLM53TransferError(str(exc)) from exc
    if report.warnings:
        raise GLM53TransferError("; ".join(report.warnings))

    config_sha = _sha256_file(config_path)
    index_sha = _sha256_file(index_path)
    files = (
        CheckpointFile("config.json", config_path.stat().st_size, config_sha),
        CheckpointFile(
            "model.safetensors.index.json",
            index_path.stat().st_size,
            index_sha,
        ),
    )
    checkpoint_hash = _stable_hash(
        {
            "revision": checkpoint_revision,
            "config_sha256": config_sha,
            "index_sha256": index_sha,
        }
    )
    manifest = DonorManifest(
        schema_version=2,
        donor_id="glm53",
        architecture_family=inspector.config.model_type,
        checkpoint_revision=checkpoint_revision,
        checkpoint_hash=checkpoint_hash,
        config_hash=_stable_hash(inspector.config.to_mapping()),
        tokenizer_hash="not-used:mamba3-foundation",
        license=donor_license,
        dtype=inspector.config.dtype,
        num_layers=inspector.config.num_hidden_layers,
        hidden_size=inspector.config.hidden_size,
        vocab_size=inspector.config.vocab_size,
        operator_layout_version="glm53-mla-dsa-moe-streaming-index-v1",
        source_uri=source_uri or root.resolve().as_uri(),
        files=files,
        tensors=(),
    )
    if manifest_output is not None:
        manifest.write_json(manifest_output)
    return GLM53DonorArtifact(manifest=manifest, checkpoint_dir=root)


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
