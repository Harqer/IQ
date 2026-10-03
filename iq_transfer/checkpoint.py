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
        self._index_path: Path | None = None
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise DonorError("invalid safetensors index: missing weight_map")
            self._weight_map = {str(k): str(v) for k, v in weight_map.items()}
            self._index_path = index_path
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

    def checkpoint_files(self) -> tuple[Path, ...]:
        files = {self.model_dir / shard for shard in self._weight_map.values()}
        if self._index_path is not None:
            files.add(self._index_path)
        return tuple(sorted(files, key=lambda p: p.name))

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

    def get_rows(self, key: str, row_indices: Any) -> Any:
        """Read only requested rows from a rank-2 safetensors tensor."""
        try:
            import torch
        except ImportError as exc:
            raise DonorError("PyTorch is required for row-sliced checkpoint reads") from exc
        indices = torch.as_tensor(row_indices, dtype=torch.long, device="cpu").reshape(-1)
        if indices.numel() == 0:
            shape = self.shape(key)
            if len(shape) != 2:
                raise DonorError(f"get_rows requires a rank-2 tensor: {key}")
            return torch.empty((0, shape[1]))
        if bool((indices < 0).any()):
            raise DonorError("row indices must be non-negative")
        shape = self.shape(key)
        if len(shape) != 2:
            raise DonorError(f"get_rows requires a rank-2 tensor: {key}")
        if int(indices.max()) >= shape[0]:
            raise DonorError(
                f"row index {int(indices.max())} exceeds tensor rows {shape[0]}"
            )

        unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
        safe_open = self._safe_open()
        pieces: list[Any] = []
        with safe_open(self._path(key), framework="pt", device="cpu") as handle:
            view = handle.get_slice(key)
            start = 0
            while start < unique.numel():
                first = int(unique[start])
                end = start + 1
                while (
                    end < unique.numel()
                    and int(unique[end]) == int(unique[end - 1]) + 1
                ):
                    end += 1
                last = int(unique[end - 1]) + 1
                pieces.append(view[first:last, :])
                start = end
        gathered_unique = torch.cat(pieces, dim=0)
        return gathered_unique[inverse].contiguous()
