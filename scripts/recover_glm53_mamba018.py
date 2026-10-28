#!/usr/bin/env python3
"""Recover a verified orphaned Mamba safetensors shard without loading donors.

Only targets the provenance-locked 20261025a run's one abandoned temporary
file. Does not modify the orphan. Default is read-only validation; explicit
--publish requires a valid safetensors layout and independent SHA256 readback.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

from huggingface_hub import download_bucket_files, get_bucket_paths_info
from safetensors import safe_open

from iq_model import HybridLayerType, IQHybridConfig
from iq_transfer.glm53_bucket_persistence import publish_file
from iq_transfer.mamba3_init import Mamba3Layout


BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "glm53-full78-local-20261025a"
PREFIX = f"IQ/runs/{RUN}"
FINAL = "model-mamba-018.safetensors"
ORPHAN = ".model-mamba-018.safetensors.mgbn0e4t.partial"


def expected_mamba_shapes(config: IQHybridConfig) -> dict[str, tuple[tuple[int, ...], str]]:
    positions = config.schedule.positions(HybridLayerType.MAMBA3)
    if len(positions) != 79:
        raise RuntimeError("Not the pinned full78 / 79-Mamba schedule")
    c = config.mamba3
    layout = Mamba3Layout(
        d_model=c.d_model, d_state=c.d_state, expand=c.expand,
        headdim=c.headdim, ngroups=1, rope_fraction=c.rope_fraction,
        is_mimo=True, mimo_rank=c.mimo_rank,
    )
    base = f"layers.{positions[18]}"
    core = f"{base}.mamba.core"
    n = layout.nheads
    return {
        f"{base}.norm.weight": ((c.d_model,), "BF16"),
        f"{core}.in_proj.weight": (layout.in_proj_shape, "BF16"),
        f"{core}.dt_bias": ((n,), "F32"),
        f"{core}.B_bias": ((n, c.mimo_rank, c.d_state), "F32"),
        f"{core}.C_bias": ((n, c.mimo_rank, c.d_state), "F32"),
        f"{core}.B_norm.weight": ((c.d_state,), "BF16"),
        f"{core}.C_norm.weight": ((c.d_state,), "BF16"),
        f"{core}.mimo_x": ((n, c.mimo_rank, c.headdim), "F32"),
        f"{core}.mimo_z": ((n, c.mimo_rank, c.headdim), "F32"),
        f"{core}.mimo_o": ((n, c.mimo_rank, c.headdim), "F32"),
        f"{core}.D": ((n,), "F32"),
        f"{core}.out_proj.weight": (layout.out_proj_shape, "BF16"),
    }


def validate_local_shard(path: Path, config: IQHybridConfig) -> None:
    expected = expected_mamba_shapes(config)
    # safetensors validates declared tensor offsets vs the actual file size
    # without materializing the 213MB tensor payload in PyTorch memory.
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        if keys != set(expected):
            raise RuntimeError(
                f"Wrong Mamba shard tensor keys; missing={sorted(set(expected)-keys)} "
                f"extra={sorted(keys-set(expected))}"
            )
        for key, (shape, dtype) in expected.items():
            item = reader.get_slice(key)
            if tuple(item.get_shape()) != tuple(shape) or item.get_dtype() != dtype:
                raise RuntimeError(f"Incorrect tensor metadata in orphan: {key}")
    print(f"IQ_MAMBA018_ORPHAN_HEADER_VALID={path.stat().st_size}", flush=True)


def _remote_info(name: str):
    infos = list(get_bucket_paths_info(BUCKET, [f"{PREFIX}/{name}"]))
    if len(infos) > 1:
        raise RuntimeError(f"Unexpected duplicate remote object: {name}")
    return infos[0] if infos else None


def recover(*, publish: bool) -> None:
    if _remote_info(FINAL) is not None:
        raise RuntimeError("Final shard already exists; refusing any overwrite")
    orphan_info = _remote_info(ORPHAN)
    if orphan_info is None or orphan_info.size < 16:
        raise RuntimeError("Original orphaned Mamba shard is missing or too small")
    if _remote_info("run-provenance.json") is None:
        raise RuntimeError("Run provenance is missing; recovery not authorized")
    with tempfile.TemporaryDirectory(prefix="iq-mamba018-recovery-") as scratch:
        td = Path(scratch)
        orphan = td / ORPHAN
        conf = td / "iq_config.json"
        download_bucket_files(
            BUCKET, files=[
                (f"{PREFIX}/{ORPHAN}", str(orphan)),
                (f"{PREFIX}/iq_config.json", str(conf)),
            ], raise_on_missing_files=True,
        )
        if orphan.stat().st_size != orphan_info.size:
            raise RuntimeError("Orphan size changed during download")
        config = IQHybridConfig.from_json(str(conf))
        validate_local_shard(orphan, config)
        digest = hashlib.sha256(orphan.read_bytes()).hexdigest()
        print(f"IQ_MAMBA018_LOCAL_SHA256={digest}", flush=True)
        if not publish:
            print("IQ_MAMBA018_VALIDATED_NO_REMOTE_CHANGE=1", flush=True)
            return
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("Set HF_TOKEN from your authorized Hugging Face login")
        output = Path.home() / "iq-hf-output" / "IQ" / "runs" / RUN
        os.environ["IQ_HF_DURABLE_OUTPUT"] = str(output)
        os.environ["IQ_HF_DURABLE_BUCKET"] = BUCKET
        os.environ["IQ_HF_DURABLE_PREFIX"] = PREFIX
        os.environ["IQ_HF_DURABLE_STAGING"] = str(td / "publish-stage")
        publish_file(output, FINAL, orphan)
        print("IQ_MAMBA018_RECOVERED_AND_REMOTE_SHA256_VERIFIED=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    recover(publish=args.publish)


if __name__ == "__main__":
    main()
