from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .checkpoint import SafetensorsSource
from .donor import DonorConfig
from .manifest import DonorManifest, build_donor_manifest
from .phi4 import Phi4Inspector


class PhiSnapshotError(RuntimeError):
    pass


_TOKENIZER_NAMES = {
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
}


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_asset_hash(model_dir: str | Path) -> str:
    root = Path(model_dir)
    candidates = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.name.startswith("tokenizer")
                or path.name in _TOKENIZER_NAMES
            )
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not candidates:
        raise PhiSnapshotError(f"no tokenizer assets found under {root}")

    digest = sha256()
    for path in candidates:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = _hash_file(path).encode("ascii")
        digest.update(file_hash)
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalPhiSnapshot:
    model_dir: Path
    config: DonorConfig
    source: SafetensorsSource
    inspector: Phi4Inspector
    manifest: DonorManifest


def open_local_phi_snapshot(
    model_dir: str | Path,
    *,
    checkpoint_revision: str,
    license: str = "MIT",
    donor_id: str = "phi",
) -> LocalPhiSnapshot:
    root = Path(model_dir).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise PhiSnapshotError(f"missing Phi config: {config_path}")
    if not checkpoint_revision.strip() or not license.strip() or not donor_id.strip():
        raise PhiSnapshotError("checkpoint_revision, license, and donor_id must be non-empty")

    config = DonorConfig.from_json(config_path)
    inspector = Phi4Inspector(config)
    source = SafetensorsSource(root)
    report = inspector.validate_checkpoint(source)
    report.require_ok()

    manifest = build_donor_manifest(
        config,
        source,
        donor_id=donor_id,
        checkpoint_revision=checkpoint_revision,
        tokenizer_hash=tokenizer_asset_hash(root),
        license=license,
        operator_layout_version="phi3-fused-v1",
        source_uri=root.as_uri(),
    )
    return LocalPhiSnapshot(root, config, source, inspector, manifest)


def load_local_phi_transformers(
    snapshot: LocalPhiSnapshot,
    *,
    torch_dtype: Any = None,
    device_map: Any = None,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise PhiSnapshotError(
            "transformers is required for activation capture; install requirements-transfer-runtime.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(snapshot.model_dir, **kwargs)
    if getattr(model.config, "model_type", None) != "phi3":
        raise PhiSnapshotError(
            f"loaded model_type changed unexpectedly: {getattr(model.config, 'model_type', None)!r}"
        )
    return model, tokenizer
