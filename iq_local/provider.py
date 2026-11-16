"""Three-tier streaming weight provider: RAM LRU, NVMe cache, HF bucket.

Task 6 of docs/IQ_LOCAL_STREAMING_PLAN.md. Tier order per expert: RAM LRU
(dequantized bf16 triple) -> NVMe int4 copy -> bucket bf16 fetch (quantized
once, int4 persisted under ``int4/<filename>``, the raw bf16 expert file is
never cached). Whole non-expert shards are cached bf16 and pinned before the
provider returns; tensors are read per-key with ``safetensors.safe_open``.

Temp-file lifecycle: every download is staged on the cache filesystem
(``staging.tmp`` for bf16, ``int4-staging.tmp`` for the saved int4 payload) so
``os.replace`` stays atomic and the cache disk guard sees staged bytes. On any
failure after download but before commit the staged file is unlinked, so no
temp leaks into the LRU accounting.

The int4 payload is stored as a flat safetensors file (keys ``<name>.packed`` /
``<name>.scales``): ``torch.load`` routes every ``.safetensors`` path to
``safetensors.load_file`` on torch >= 2.6, so a ``torch.save`` payload under
the prescribed ``int4/<expert>.safetensors`` key could never be read back.
Output shapes are derived from the packed dims (cols = packed cols * 2).

Not thread-safe: the staging names are single-flight per provider and the RAM
LRU is unsynchronized. The Phase-1 server serves one request at a time;
callers must serialize use of one provider instance.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from iq_local.cache import LocalCache
from iq_local.index import RunIndex
from iq_local.quant import PackedExpert, dequantize_int4, quantize_int4

_STAGING_TMP = "staging.tmp"
_INT4_STAGING_TMP = "int4-staging.tmp"


class StreamingWeightProvider:
    def __init__(
        self,
        index: RunIndex,
        cache: LocalCache,
        client,
        *,
        max_ram_experts: int = 64,
    ) -> None:
        self.index = index
        self.cache = cache
        self.client = client
        self.max_ram_experts = max_ram_experts
        self._ram: OrderedDict[str, tuple[torch.Tensor, ...]] = OrderedDict()
        self._stage_layers: dict[int, int] = {}
        self.counters = {
            "bucket_files": 0,
            "bucket_bytes": 0,
            "nvme_hits": 0,
            "ram_hits": 0,
            "int4_converted": 0,
        }

    # -- shard tier ---------------------------------------------------------

    def _shard_path(self, filename: str) -> Path:
        key = f"shards/{filename}"
        if self.cache.contains(key):
            self.counters["nvme_hits"] += 1
            path = self.cache.path_for(key)
        else:
            path = self._fetch_shard(filename, key)
        self.cache.pin(key)
        return path

    def _fetch_shard(self, filename: str, key: str) -> Path:
        tmp = self.cache.path_for(_STAGING_TMP)
        if tmp.exists():
            tmp.unlink()
        try:
            self.client.download([(filename, tmp)])
        except BaseException:
            self._unlink(tmp)
            raise
        self.counters["bucket_files"] += 1
        self.counters["bucket_bytes"] += tmp.stat().st_size
        try:
            return self.cache.commit(key, tmp)
        except BaseException:
            self._unlink(tmp)
            raise

    def _read_keys(self, path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in keys:
                out[key] = handle.get_tensor(key)
        return out

    def load_global(self, key: str) -> torch.Tensor:
        """Return one bf16 tensor by state-dict key, via the shard cache tier."""
        filename = self.index.weight_map[key]
        path = self._shard_path(filename)
        return self._read_keys(path, [key])[key]

    def load_layer_base(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return every layer key that is not ``moe.routed_experts.*``."""
        prefix = f"layers.{layer_idx}."
        excluded = f"{prefix}moe.routed_experts."
        keys = [
            key
            for key in self.index.weight_map
            if key.startswith(prefix) and not key.startswith(excluded)
        ]
        by_file: dict[str, list[str]] = {}
        for key in keys:
            by_file.setdefault(self.index.weight_map[key], []).append(key)
        tensors: dict[str, torch.Tensor] = {}
        for filename, file_keys in by_file.items():
            path = self._shard_path(filename)
            tensors.update(self._read_keys(path, file_keys))
        return tensors

    # -- expert tier --------------------------------------------------------

    def _expert_key(self, stage: int, expert_idx: int) -> str:
        return f"{stage}:{expert_idx:03d}"

    def _expert_filename(self, stage: int, expert_idx: int) -> str:
        return f"model-moe-{stage:03d}-expert-{expert_idx:03d}.safetensors"

    def _ram_get(self, key: str):
        if key not in self._ram:
            return None
        self._ram.move_to_end(key)
        self.counters["ram_hits"] += 1
        return self._ram[key]

    def _ram_put(self, key: str, triple: tuple[torch.Tensor, ...]) -> None:
        self._ram[key] = triple
        self._ram.move_to_end(key)
        while len(self._ram) > self.max_ram_experts:
            self._ram.popitem(last=False)

    def load_expert(
        self, stage: int, expert_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dequantized bf16 ``(gate, up, down)`` for one routed expert."""
        key = self._expert_key(stage, expert_idx)
        cached = self._ram_get(key)
        if cached is not None:
            return cached
        filename = self._expert_filename(stage, expert_idx)
        int4_key = f"int4/{filename}"
        if self.cache.contains(int4_key):
            self.cache.pin(int4_key)
            payload = torch.load(
                self.cache.path_for(int4_key), map_location="cpu", weights_only=True
            )
            triple = self._dequant_payload(payload)
        else:
            triple = self._acquire_expert(stage, expert_idx, filename, int4_key)
        self._ram_put(key, triple)
        return triple

    def _acquire_expert(
        self, stage: int, expert_idx: int, filename: str, int4_key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmp = self.cache.path_for(_STAGING_TMP)
        if tmp.exists():
            tmp.unlink()
        try:
            self.client.download([(filename, tmp)])
        except BaseException:
            self._unlink(tmp)
            raise
        self.counters["bucket_files"] += 1
        self.counters["bucket_bytes"] += tmp.stat().st_size
        try:
            packed = self._quantize_staged(tmp, stage, expert_idx, filename)
        finally:
            self._unlink(tmp)
        payload = {}
        for name, expert in packed.items():
            payload[f"{name}.packed"] = expert.packed
            payload[f"{name}.scales"] = expert.scales
        staged = self.cache.path_for(_INT4_STAGING_TMP)
        try:
            save_file(payload, str(staged))
            self.cache.commit(int4_key, staged)
        except BaseException:
            self._unlink(staged)
            raise
        self.counters["int4_converted"] += 1
        self.cache.pin(int4_key)
        return self._dequant_payload(payload)

    def _quantize_staged(
        self, tmp: Path, stage: int, expert_idx: int, filename: str
    ) -> dict[str, PackedExpert]:
        layer = self._expert_layer(stage)
        keys = list(self.index.keys_for_file(filename))
        raw = None
        try:
            with safe_open(str(tmp), framework="pt", device="cpu") as handle:
                raw = {key: handle.get_tensor(key).to(torch.float32) for key in keys}
            return {
                name: quantize_int4(
                    raw[f"layers.{layer}.moe.routed_experts.{expert_idx}.{proj}.weight"]
                )
                for name, proj in (
                    ("gate", "gate_proj"),
                    ("up", "up_proj"),
                    ("down", "down_proj"),
                )
            }
        finally:
            del raw

    def _expert_layer(self, stage: int) -> int:
        if stage not in self._stage_layers:
            sample = self.index.experts_by_stage[stage][0]
            first_key = self.index.keys_for_file(sample)[0]
            self._stage_layers[stage] = int(first_key.split(".")[1])
        return self._stage_layers[stage]

    @staticmethod
    def _dequant_payload(payload) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = []
        for name in ("gate", "up", "down"):
            packed = payload[f"{name}.packed"]
            out.append(
                dequantize_int4(
                    PackedExpert(
                        packed=packed,
                        scales=payload[f"{name}.scales"],
                    ),
                    (packed.shape[0], packed.shape[1] * 2),
                    torch.bfloat16,
                )
            )
        return tuple(out)

    @staticmethod
    def _unlink(path: Path) -> None:
        if path.exists():
            path.unlink()
