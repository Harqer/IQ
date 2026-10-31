#!/usr/bin/env python3
"""Independent, bounded numerical spot checks for a persisted IQ full78 transfer.

Read-only: never writes the bucket, recompiles weights, loads full donor model,
or launches inference. Checks:
* sparse stage-003 expert-000 gate/up/down rows against original GLM BF16 donor
  rows multiplied by the saved WARM latent basis;
* every data byte of dense stage-000 routed expert-000 is zero;
* Mamba-018 saved norm, D, dt_bias and B_bias obey exact 2x widening.

Uses verified HTTP 206 ranges for specific tensor bytes, NOT the unreliable
NFS mount. Numerical checks do not establish full-model functional fidelity.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np
import requests
import torch
from huggingface_hub import HfFileSystem, get_bucket_paths_info

from scripts.finalize_glm53_full78_metadata import (
    BUCKET, RUN, GLM_REV, MAMBA_REV, MAX_HEADER,
    checkpoint_json, expected_names,
)
from scripts.probe_glm53_remote_ranges import fetch_exact_range
from iq_transfer.http_range_source import HubSafetensorsSource
from iq_transfer.glm53_lazy_calibration import LazyGLM53Calibration

BYTE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8}
TORCH_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I32": torch.int32, "I64": torch.int64,
}
MAX_CHUNK = 2 * 1024 * 1024
SPARSE_STAGE, SPARSE_SLOT = 3, 0
DENSE_STAGE, DENSE_SLOT = 0, 0
MAMBA_ORDINAL = 18


def verify_contiguous_header(data: dict[str, Any], remote_size: int,
                             header_size: int) -> None:
    if not isinstance(data, dict) or not data or not 0 <= header_size < remote_size:
        raise RuntimeError("Invalid remote safetensors header")
    end = 0
    spans = []
    for key, entry in data.items():
        if key == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid tensor entry {key}")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        off = entry.get("data_offsets")
        if (not isinstance(shape, list) or
            # Scalars are represented by shape [] and occupy one element.
            any(type(n) is not int or n < 1 for n in shape) or
            dtype not in BYTE_SIZES or
            not isinstance(off, list) or len(off) != 2 or
            any(type(n) is not int for n in off)):
            raise RuntimeError(f"Invalid safetensors metadata: {key}")
        count = math.prod(shape) * BYTE_SIZES[dtype]
        a, b = off
        if a < 0 or b - a != count:
            raise RuntimeError(f"Wrong tensor byte count: {key}")
        spans.append((a, b))
    if not spans:
        raise RuntimeError("No tensors in safetensors header")
    for a, b in sorted(spans):
        if a != end:
            raise RuntimeError("Overlapping or gapped tensor data")
        end = b
    if end + 8 + header_size != remote_size:
        raise RuntimeError("Tensor data does not cover remote shard exactly")


def pass_compare(expected: np.ndarray, actual: np.ndarray,
                 *, rtol: float = 0.025, atol: float = 0.0015) -> dict:
    """Compare against BF16-rounded independent reference with error metrics.

    The compiler used FP32 CUDA matmul while the verifier uses FP64 CPU.
    Exact BF16 equality is informative but not always expected at rounding
    boundaries. Differences beyond the tolerance fail closed.
    """
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError(f"Bad numerical array shapes/values: {expected.shape} vs {actual.shape}")
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))
    failures = int(np.count_nonzero(delta > allowed))
    stats = {
        "elements": int(expected.size),
        "bad": failures,
        "max_abs": float(np.max(delta)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "bf16_exact_fraction": float(np.mean(expected == actual)),
    }
    if failures:
        raise RuntimeError(f"Numerical mismatch: {stats}")
    return stats


class RemoteShard:
    """Per-shard verified HTTP range reader; never fetches whole shard by default."""

    def __init__(self, fs: HfFileSystem, session: requests.Session,
                 name: str, remote_size: int):
        self.name, self.size, self.session = name, int(remote_size), session
        uri = f"hf://buckets/{BUCKET}/IQ/runs/{RUN}/{name}"
        self.url = fs.url(uri)
        self.token = os.environ.get("HF_TOKEN")
        self.header_size, self.entries = self._load_header()

    def _range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise RuntimeError(f"Range outside remote object {self.name}")
        length = end - start + 1
        if length > MAX_CHUNK:
            raise RuntimeError("Range exceeds bounded 2 MiB validation limit")
        chunk, length_on_server = fetch_exact_range(
            self.session, self.url, start, end, token=self.token
        )
        if length_on_server != self.size or len(chunk) != length:
            raise RuntimeError(f"Remote shard changed during numerical check: {self.name}")
        return chunk

    def _load_header(self) -> tuple[int, dict]:
        leading = self._range(0, 7)
        header_size = struct.unpack("<Q", leading)[0]
        if not 2 <= header_size <= MAX_HEADER:
            raise RuntimeError(f"Unreasonable header length: {self.name}")
        chunks = [
            self._range(a, min(a + MAX_CHUNK - 1, 8 + header_size - 1))
            for a in range(8, 8 + header_size, MAX_CHUNK)
        ]
        header = json.loads(b"".join(chunks))
        verify_contiguous_header(header, self.size, header_size)
        header.pop("__metadata__", None)
        return header_size, header

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.entries[key]["shape"])

    def tensor_row(self, key: str, row: int) -> np.ndarray:
        entry = self.entries[key]
        if len(entry["shape"]) != 2:
            raise RuntimeError("Requested row of nonmatrix")
        height, width = entry["shape"]
        if not 0 <= row < height:
            raise RuntimeError("Invalid sampled output row")
        itembytes = BYTE_SIZES[entry["dtype"]]
        begin = 8 + self.header_size + entry["data_offsets"][0]
        start = begin + row * width * itembytes
        length = width * itembytes
        if length > MAX_CHUNK:
            raise RuntimeError("Sample row too wide")
        return self._as_float(self._range(start, start + length - 1), entry["dtype"])

    @staticmethod
    def _as_float(payload: bytes, dtype: str) -> np.ndarray:
        if dtype not in TORCH_DTYPES:
            raise RuntimeError(f"Unsupported value dtype {dtype}")
        # Bytearray owns the buffer; clone makes the torch tensor independent.
        data = torch.frombuffer(bytearray(payload), dtype=TORCH_DTYPES[dtype]).clone()
        return data.to(torch.float32).numpy()

    def tensor(self, key: str, *, max_bytes: int = MAX_CHUNK) -> np.ndarray:
        entry = self.entries[key]
        a, b = entry["data_offsets"]
        if b - a > max_bytes:
            raise RuntimeError(f"Requested large tensor in small-tensor check: {key}")
        base = 8 + self.header_size
        value = self._as_float(self._range(base + a, base + b - 1), entry["dtype"])
        return value.reshape(entry["shape"])

    def data_is_all_zero(self) -> tuple[int, int]:
        """Check full tensor data bytes; safetensors BF16 zeros are all 0x00."""
        start = 8 + self.header_size
        bytes_checked, nonzero = 0, 0
        while start < self.size:
            end = min(self.size - 1, start + MAX_CHUNK - 1)
            chunk = self._range(start, end)
            bytes_checked += len(chunk)
            nonzero += len(chunk) - chunk.count(0)
            if nonzero:
                raise RuntimeError(f"Expected zero-initialized dense expert, found nonzero data: {self.name}")
            start = end + 1
        return bytes_checked, nonzero


def verify_dense_zero(shard: RemoteShard, *, physical: int) -> dict:
    expected = {
        f"layers.{physical}.moe.routed_experts.{DENSE_SLOT}.{kind}_proj.weight"
        for kind in ("gate", "up", "down")
    }
    if set(shard.entries) != expected or any(
        entry["dtype"] != "BF16" for entry in shard.entries.values()
    ):
        raise RuntimeError("Dense expert tensor layout mismatch")
    checked, nonzero = shard.data_is_all_zero()
    result = {"bytes_checked": checked, "nonzero_bytes": nonzero}
    print(f"IQ_NUMERIC_DENSE_ZERO_PASS={shard.name} {result}", flush=True)
    return result


def verify_mamba_widening(shard: RemoteShard, *, physical: int) -> dict:
    prefix = f"layers.{physical}."
    suffixes = ("norm.weight", "mamba.core.dt_bias",
                "mamba.core.D", "mamba.core.B_bias")
    stats = {}
    for suffix in suffixes:
        key = prefix + suffix
        arr = shard.tensor(key)
        if arr.shape[0] < 2 or arr.shape[0] % 2:
            raise RuntimeError(f"Not a 2x-replicated Mamba tensor: {key}")
        left, right = np.split(arr, 2, axis=0)
        if not np.array_equal(left, right):
            raise RuntimeError(f"Mamba replication mismatch: {key}")
        stats[suffix] = int(arr.size)
    print(f"IQ_NUMERIC_MAMBA_REPLICATION_PASS={shard.name} {stats}", flush=True)
    return stats


def compute_sparse_row_reference(
    source_row: np.ndarray, latent_map: np.ndarray
) -> np.ndarray:
    """Independently evaluate donor row @ P_latent in FP64 then round BF16."""
    if source_row.ndim != 1 or latent_map.ndim != 2 or source_row.shape[0] != latent_map.shape[0]:
        raise RuntimeError("Sparse projection geometry mismatch")
    if not np.isfinite(source_row).all():
        raise RuntimeError("Nonfinite donor row")
    # The compiler proves this basis's columns are orthonormal; for these
    # canonical maps the pseudoinverse-transpose equals P_latent itself.
    reference = source_row.astype(np.float64) @ latent_map
    return torch.from_numpy(np.ascontiguousarray(reference)).to(torch.bfloat16).float().numpy()


def verify_sparse(
    shard: RemoteShard, *, root: Path, stage: int, slot: int,
    physical: int, report: dict,
) -> dict:
    calib = LazyGLM53Calibration(root / "calibration").stage(stage)
    if calib.source_layer != stage or calib.moe_physical_layer != physical or calib.expert_usage is None:
        raise RuntimeError("Sparse calibration stage mismatch")
    scores = np.asarray(calib.expert_usage, dtype=np.float64)
    if scores.shape != (256,) or not np.isfinite(scores).all() or (scores < 0).any():
        raise RuntimeError("Invalid expert usage")
    order = np.lexsort((np.arange(256), -scores))
    donor_id = int(order[slot])
    routed = report.get("routed", {})
    record = routed.get(str(slot))
    if not isinstance(record, dict) or record.get("donor_expert_id") != donor_id:
        raise RuntimeError(f"Selected donor expert does not match saved report: expected {donor_id}")
    latent = np.asarray(calib.latent_map.matrix)
    if latent.shape != (6144, 2048):
        raise RuntimeError(f"Unexpected GLM latent projection shape: {latent.shape}")

    # Verify the prerequisite for using P rather than a costly pseudoinverse.
    # Narrow Gram probe; do not allocate an enormous full Gram matrix.
    columns = [0, 17, 511, 1024, 2047]
    gram = latent.T @ latent[:, columns]
    for j, col in enumerate(columns):
        expected = np.zeros(latent.shape[1])
        expected[col] = 1.
        if not np.allclose(gram[:, j], expected, atol=1e-5, rtol=0):
            raise RuntimeError(f"Nonorthogonal calibration basis at column {col}")

    donor_root = root / "donors" / "glm53"
    source = HubSafetensorsSource(
        donor_root, repo_id="zai-org/GLM-5.3-BF16",
        revision=GLM_REV, max_request_bytes=2 * 1024 * 1024,
        max_tensor_bytes=32 * 1024 * 1024,
    )
    result = {}
    try:
        for kind in ("gate", "up"):
            donor_key = f"model.layers.{stage}.mlp.experts.{donor_id}.{kind}_proj.weight"
            target_key = f"layers.{physical}.moe.routed_experts.{slot}.{kind}_proj.weight"
            if source.shape(donor_key) != (2048, 6144):
                raise RuntimeError(f"Wrong original donor tensor shape: {donor_key}")
            if shard.tensor_shape(target_key) != (2048, 2048):
                raise RuntimeError(f"Wrong saved tensor shape: {target_key}")
            summary = []
            for r in (0, 37, 1024, 2047):
                donor = source.get_rows(donor_key, [r]).float().numpy()[0]
                expected = compute_sparse_row_reference(donor, latent)
                actual = shard.tensor_row(target_key, r)
                summary.append(pass_compare(expected, actual))
            result[kind] = {
                "rows": 4,
                "max_abs": max(x["max_abs"] for x in summary),
                "bad": sum(x["bad"] for x in summary),
                "mean_bf16_exact_fraction": float(
                    np.mean([x["bf16_exact_fraction"] for x in summary])
                ),
            }
            print(f"IQ_NUMERIC_SPARSE_ROW_PASS=stage:{stage} slot:{slot} donor:{donor_id} "
                  f"tensor:{kind} {result[kind]}", flush=True)

        # Full donor down is a single 25 MiB BF16 tensor. The compiler
        # projects it with P_latent.T @ W_down. Read once, then verify four
        # target rows independently in FP64 without constructing the full
        # 2048x2048 projected matrix.
        donor_key = f"model.layers.{stage}.mlp.experts.{donor_id}.down_proj.weight"
        target_key = f"layers.{physical}.moe.routed_experts.{slot}.down_proj.weight"
        if (source.shape(donor_key) != (6144, 2048) or
            shard.tensor_shape(target_key) != (2048, 2048)):
            raise RuntimeError("Wrong sparse donor/target down-projection geometry")
        donor_down = source.get(donor_key).float().numpy().astype(np.float64)
        summary = []
        for r in (0, 37, 1024, 2047):
            reference = latent[:, r] @ donor_down
            expected = torch.from_numpy(
                np.ascontiguousarray(reference)
            ).to(torch.bfloat16).float().numpy()
            actual = shard.tensor_row(target_key, r)
            summary.append(pass_compare(expected, actual))
        result["down"] = {
            "rows": 4,
            "max_abs": max(x["max_abs"] for x in summary),
            "bad": sum(x["bad"] for x in summary),
            "mean_bf16_exact_fraction": float(
                np.mean([x["bf16_exact_fraction"] for x in summary])
            ),
        }
        print(f"IQ_NUMERIC_SPARSE_ROW_PASS=stage:{stage} slot:{slot} donor:{donor_id} "
              f"tensor:down {result['down']}", flush=True)
    finally:
        source.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.home() / "IQ-transfer-full78",
                   help="Existing local donor metadata and calibrated maps")
    p.add_argument("--only", choices=("all", "sparse", "dense", "mamba"),
                   default="all")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Optional local report path (no bucket writes)")
    a = p.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN required for authenticated bucket and donor range reads")
    root = a.root
    cal_meta = json.loads((root / "calibration/glm53_calibration.json").read_text())
    schedule = json.loads(
        (root / "donors/glm53/config.json").read_text()
    )
    if len(cal_meta["stages"]) != 78 or len(schedule.get("model_type", "")) < 1:
        raise RuntimeError("Missing verified pinned donor/calibration metadata")

    fs = HfFileSystem(token=os.environ["HF_TOKEN"])
    # Config is on HF bucket, read via API (not stale NFS).
    output = Path("nonexistent-read-only-placeholder")
    config = checkpoint_json(fs, output, "iq_config.json", read_mode="hub")
    layers = config["schedule"]["layers"]
    records = cal_meta["stages"]
    sparse_physical = int(records[SPARSE_STAGE]["moe_physical_layer"])
    dense_physical = int(records[DENSE_STAGE]["moe_physical_layer"])
    mamba_physical = [i for i, k in enumerate(layers) if k == "mamba3"][MAMBA_ORDINAL]
    if (config["stable_moe"]["num_experts"] != 256 or
        layers[sparse_physical] != "moe" or
        layers[dense_physical] != "moe" or len(layers) != 235):
        raise RuntimeError("Mismatched saved IQ schedule")

    specs = {}
    if a.only in ("all", "sparse"):
        specs["sparse"] = f"model-moe-{SPARSE_STAGE:03d}-expert-{SPARSE_SLOT:03d}.safetensors"
    if a.only in ("all", "dense"):
        specs["dense"] = f"model-moe-{DENSE_STAGE:03d}-expert-{DENSE_SLOT:03d}.safetensors"
    if a.only in ("all", "mamba"):
        specs["mamba"] = f"model-mamba-{MAMBA_ORDINAL:03d}.safetensors"
    prefix = f"IQ/runs/{RUN}/"
    entries = {
        info.path.removeprefix(prefix): info.size
        for info in get_bucket_paths_info(
            BUCKET, [prefix + name for name in specs.values()]
        )
    }
    if set(entries) != set(specs.values()):
        raise RuntimeError(f"Missing validation shard(s) on HF: {set(specs.values())-set(entries)}")

    result = {"run": RUN, "scope": a.only, "checks": {}}
    with requests.Session() as session:
        for label, name in specs.items():
            shard = RemoteShard(fs, session, name, entries[name])
            if label == "dense":
                result["checks"]["dense"] = verify_dense_zero(
                    shard, physical=dense_physical,
                )
            elif label == "mamba":
                result["checks"]["mamba"] = verify_mamba_widening(
                    shard, physical=mamba_physical,
                )
            elif label == "sparse":
                report = checkpoint_json(
                    fs, output, f"model-moe-{SPARSE_STAGE:03d}-projection.json",
                    read_mode="hub",
                )
                result["checks"]["sparse"] = verify_sparse(
                    shard, root=root, stage=SPARSE_STAGE,
                    slot=SPARSE_SLOT, physical=sparse_physical, report=report,
                )
    print("IQ_NUMERIC_SAMPLES_PASS=" + ",".join(result["checks"]), flush=True)
    if a.output_json:
        if a.output_json.exists():
            raise RuntimeError(f"Refuse to overwrite existing verification result {a.output_json}")
        a.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"IQ_NUMERIC_REPORT_SAVED={a.output_json}", flush=True)


if __name__ == "__main__":
    main()
