"""Run-side index: fetch run manifests, parse weight_map, build expert map."""

from __future__ import annotations

import json
import re
from pathlib import Path

from iq_local.bucket import BucketClient

_MANIFESTS = (
    "iq_config.json",
    "model.safetensors.index.json",
    "COMPILE_COMPLETE.json",
    "transplant_manifest.json",
)

_EXPERT_FILE = re.compile(r"^model-moe-(\d{3})-expert-(\d{3})\.safetensors$")


class RunIndex:
    def __init__(self, client: BucketClient | None, work_dir: Path) -> None:
        self.client = client
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for name in _MANIFESTS:
            self._ensure(name)
        self.config = self._load_config()
        index_payload = json.loads(
            (self.work_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.weight_map: dict[str, str] = dict(index_payload["weight_map"])
        self.files: list[str] = sorted(set(self.weight_map.values()))
        self._build_experts_by_stage()
        self._stage_cache: dict[int, int] = {}

    def _ensure(self, name: str) -> None:
        dest = self.work_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            return
        if self.client is None:
            raise FileNotFoundError(f"{dest} missing and no bucket client configured")
        self.client.download([(name, dest)])

    def _load_config(self):
        from iq_model.hybrid import IQHybridConfig

        return IQHybridConfig.from_json(str(self.work_dir / "iq_config.json"))

    def _build_experts_by_stage(self) -> None:
        experts_by_stage: dict[int, list[str]] = {}
        for filename in self.files:
            match = _EXPERT_FILE.match(filename)
            if match:
                experts_by_stage.setdefault(int(match.group(1)), []).append(filename)
        self.experts_by_stage = experts_by_stage

    def stage_of_layer(self, layer_idx: int) -> int:
        """Resolve MoE stage for a schedule layer via one routed-expert key.

        Keys: `layers.<layer_idx>.moe.routed_experts.<E>...`; files:
        `model-moe-<stage>-expert-<E>.safetensors`. Memoized per layer.
        """
        if layer_idx in self._stage_cache:
            return self._stage_cache[layer_idx]
        prefix = f"layers.{layer_idx}.moe.routed_experts."
        for key, filename in self.weight_map.items():
            if key.startswith(prefix):
                match = _EXPERT_FILE.match(filename)
                if match:
                    stage = int(match.group(1))
                    self._stage_cache[layer_idx] = stage
                    return stage
                break
        raise KeyError(f"layer {layer_idx} has no routed expert keys")

    def keys_for_file(self, filename: str) -> list[str]:
        return sorted(key for key, file in self.weight_map.items() if file == filename)

    @staticmethod
    def verify_against_local_checkpoint(run_index: RunIndex, local_dir: Path) -> None:
        local_dir = Path(local_dir)
        local_config = (local_dir / "iq_config.json").read_bytes()
        run_config = (run_index.work_dir / "iq_config.json").read_bytes()
        if local_config != run_config:
            raise ValueError("run iq_config.json differs from local checkpoints/iq copy")
        local_index = json.loads(
            (local_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        if dict(local_index["weight_map"]) != run_index.weight_map:
            raise ValueError("run weight_map differs from local checkpoints/iq index")
        manifest = json.loads(
            (run_index.work_dir / "transplant_manifest.json").read_text(encoding="utf-8")
        )
        # Schema v2 keeps the recipient fingerprint at the top level
        # (`recipient_config_fingerprint`); fall back to the nested path.
        fingerprint = manifest.get("recipient_config_fingerprint", "")
        if not fingerprint:
            fingerprint = manifest.get("recipient", {}).get("config_fingerprint", "")
        if fingerprint and not str(fingerprint).startswith("6c292b47"):
            raise ValueError(f"unexpected recipient config fingerprint: {fingerprint}")
