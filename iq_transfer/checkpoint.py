from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .donor import DonorError


class SafetensorsSource:
    """Lazy local safetensors source. Shards are opened only when a tensor is requested."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise DonorError(f"checkpoint directory does not exist: {self.model_dir}")

        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise DonorError("invalid safetensors index: missing weight_map")
            self._weight_map = {str(k): str(v) for k, v in weight_map.items()}
        elif single_path.exists():
            self._weight_map = self._scan_single(single_path)
        else:
            raise DonorError("no model.safetensors or model.safetensors.index.json found")

    @staticmethod
    def _safe_open():
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise DonorError("safetensors is required to read donor checkpoints") from exc
        return safe_open

    def _scan_single(self, path: Path) -> dict[str, str]:
        safe_open = self._safe_open()
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {key: path.name for key in handle.keys()}

    def keys(self) -> tuple[str, ...]:
        return tuple(self._weight_map)

    def _path(self, key: str) -> Path:
        try:
            shard = self._weight_map[key]
        except KeyError as exc:
            raise DonorError(f"unknown checkpoint tensor: {key}") from exc
        return self.model_dir / shard

    def shape(self, key: str) -> tuple[int, ...]:
        safe_open = self._safe_open()
        with safe_open(self._path(key), framework="pt", device="cpu") as handle:
            return tuple(int(x) for x in handle.get_slice(key).get_shape())

    def get(self, key: str) -> Any:
        safe_open = self._safe_open()
        with safe_open(self._path(key), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)
