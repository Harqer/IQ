from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    step: int
    consumed_tokens: int
    model_config_hash: str
    files: tuple[tuple[str, str], ...]


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().contiguous().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config_hash: str,
    step: int,
    consumed_tokens: int,
    scheduler: Any | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    if step < 0 or consumed_tokens < 0:
        raise CheckpointError("step and consumed_tokens must be non-negative")
    if not model_config_hash.strip():
        raise CheckpointError("model_config_hash must be non-empty")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        save_file(_cpu_model_state(model), str(temp / "model.safetensors"))
        torch.save(optimizer.state_dict(), temp / "optimizer.pt")
        training_state = {
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "extra_state": extra_state or {},
        }
        torch.save(training_state, temp / "training_state.pt")

        payload_files = ("model.safetensors", "optimizer.pt", "training_state.pt")
        hashes = tuple((name, _sha256(temp / name)) for name in payload_files)
        metadata = CheckpointMetadata(
            schema_version=1,
            step=int(step),
            consumed_tokens=int(consumed_tokens),
            model_config_hash=model_config_hash,
            files=hashes,
        )
        (temp / "manifest.json").write_text(
            json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (temp / "COMPLETE").write_text("ok\n", encoding="utf-8")

        if destination.exists():
            backup = destination.with_name(destination.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
            try:
                os.replace(temp, destination)
            except Exception:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temp, destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_model_config_hash: str,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[CheckpointMetadata, dict[str, Any]]:
    source = Path(path)
    if not (source / "COMPLETE").is_file():
        raise CheckpointError(f"checkpoint is incomplete: {source}")
    try:
        raw = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("checkpoint manifest is missing or invalid") from exc
    metadata = CheckpointMetadata(
        schema_version=int(raw["schema_version"]),
        step=int(raw["step"]),
        consumed_tokens=int(raw["consumed_tokens"]),
        model_config_hash=str(raw["model_config_hash"]),
        files=tuple((str(name), str(digest)) for name, digest in raw["files"]),
    )
    if metadata.schema_version != 1:
        raise CheckpointError(f"unsupported checkpoint schema: {metadata.schema_version}")
    if metadata.model_config_hash != expected_model_config_hash:
        raise CheckpointError("checkpoint model config hash does not match the recipient configuration")
    for name, digest in metadata.files:
        file_path = source / name
        if not file_path.is_file() or _sha256(file_path) != digest:
            raise CheckpointError(f"checkpoint payload hash mismatch: {name}")

    model_state = load_file(str(source / "model.safetensors"), device=str(map_location))
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise CheckpointError(f"model state mismatch: missing={missing}, unexpected={unexpected}")

    optimizer_state = torch.load(source / "optimizer.pt", map_location=map_location, weights_only=False)
    optimizer.load_state_dict(optimizer_state)
    training_state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
    if scheduler is not None:
        if training_state["scheduler"] is None:
            raise CheckpointError("checkpoint has no scheduler state")
        scheduler.load_state_dict(training_state["scheduler"])
    elif training_state["scheduler"] is not None:
        raise CheckpointError("checkpoint contains scheduler state but no scheduler was supplied")

    torch.set_rng_state(training_state["cpu_rng_state"])
    if torch.cuda.is_available() and training_state["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
    return metadata, dict(training_state["extra_state"])
