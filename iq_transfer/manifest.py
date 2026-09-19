from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

from .donor import DonorConfig, TensorSource


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TensorInventoryItem:
    key: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DonorManifest:
    schema_version: int
    donor_id: str
    architecture_family: str
    checkpoint_revision: str
    checkpoint_hash: str
    config_hash: str
    tokenizer_hash: str
    license: str
    dtype: str | None
    num_layers: int
    hidden_size: int
    vocab_size: int | None
    operator_layout_version: str
    source_uri: str
    files: tuple[CheckpointFile, ...]
    tensors: tuple[TensorInventoryItem, ...]

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "DonorManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["files"] = tuple(CheckpointFile(**item) for item in data["files"])
        data["tensors"] = tuple(
            TensorInventoryItem(key=item["key"], shape=tuple(item["shape"])) for item in data["tensors"]
        )
        return cls(**data)


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash_mapping(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(payload).hexdigest()


def _checkpoint_hash(files: Iterable[CheckpointFile], tensors: tuple[TensorInventoryItem, ...]) -> str:
    file_list = tuple(files)
    if file_list:
        data = [{"name": x.name, "size_bytes": x.size_bytes, "sha256": x.sha256} for x in file_list]
    else:
        data = [{"key": x.key, "shape": list(x.shape)} for x in tensors]
    return _stable_hash_mapping({"inventory": data})


def build_donor_manifest(
    config: DonorConfig,
    source: TensorSource,
    *,
    donor_id: str,
    checkpoint_revision: str,
    tokenizer_hash: str,
    license: str,
    operator_layout_version: str,
    source_uri: str,
    allow_metadata_only: bool = False,
) -> DonorManifest:
    required_text = {
        "donor_id": donor_id,
        "checkpoint_revision": checkpoint_revision,
        "tokenizer_hash": tokenizer_hash,
        "license": license,
        "operator_layout_version": operator_layout_version,
        "source_uri": source_uri,
    }
    empty = [name for name, value in required_text.items() if not str(value).strip()]
    if empty:
        raise ManifestError(f"manifest fields must be non-empty: {', '.join(empty)}")

    tensors = tuple(
        TensorInventoryItem(key=key, shape=tuple(int(x) for x in source.shape(key)))
        for key in sorted(source.keys())
    )
    if not tensors:
        raise ManifestError("checkpoint tensor inventory is empty")

    checkpoint_paths = getattr(source, "checkpoint_files", None)
    files: tuple[CheckpointFile, ...] = ()
    if callable(checkpoint_paths):
        records: list[CheckpointFile] = []
        for path_like in checkpoint_paths():
            path = Path(path_like)
            if not path.is_file():
                raise ManifestError(f"checkpoint file does not exist: {path}")
            records.append(CheckpointFile(path.name, path.stat().st_size, _hash_file(path)))
        files = tuple(sorted(records, key=lambda x: x.name))
    elif not allow_metadata_only:
        raise ManifestError("production donor manifests require a source exposing checkpoint_files()")

    config_hash = _stable_hash_mapping(config.to_mapping())
    return DonorManifest(
        schema_version=1,
        donor_id=donor_id,
        architecture_family=config.model_type,
        checkpoint_revision=checkpoint_revision,
        checkpoint_hash=_checkpoint_hash(files, tensors),
        config_hash=config_hash,
        tokenizer_hash=tokenizer_hash,
        license=license,
        dtype=config.dtype,
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        operator_layout_version=operator_layout_version,
        source_uri=source_uri,
        files=files,
        tensors=tensors,
    )
