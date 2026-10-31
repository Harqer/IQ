#!/usr/bin/env python3
"""Finalize an ALREADY COMPLETE GLM53 full78 run without loading model weights.

Strictly checks all expected shards and safetensors headers one at a time.
No CUDA, no matrix inverses, no donor weight loading. Default is read-only.
--publish publishes immutable manifest, index, and then non-inference-ready
completion marker using the independently SHA256-verified HF bucket API.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "glm53-full78-local-20261025a"
GLM_REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAMBA_REV = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
MAX_HEADER = 4 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_safetensors_header(stream, file_size: int, name: str) -> dict[str, dict]:
    """Inspect exact serialized tensor spans from a bounded header-only stream."""
    if file_size < 16:
        raise RuntimeError(f"Invalid checkpoint object {name}")
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise RuntimeError(f"Short safetensors prefix {name}")
    size = struct.unpack("<Q", prefix)[0]
    if not 2 <= size <= MAX_HEADER or size + 8 > file_size:
        raise RuntimeError(f"Invalid safetensors header length {name}")
    payload = stream.read(size)
    if len(payload) != size:
        raise RuntimeError(f"Truncated safetensors header {name}")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid safetensors JSON {name}")
    data.pop("__metadata__", None)
    if not data:
        raise RuntimeError(f"Empty safetensors shard {name}")
    spans = []
    for key, item in data.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            raise RuntimeError(f"Invalid safetensors entry {name}")
        shape, dtype, offsets = item.get("shape"), item.get("dtype"), item.get("data_offsets")
        if (not isinstance(shape, list) or not shape or
            any(type(dim) is not int or dim < 1 for dim in shape) or
            dtype not in DTYPE_BYTES or not isinstance(offsets, list) or
            len(offsets) != 2 or any(type(v) is not int for v in offsets)):
            raise RuntimeError(f"Invalid tensor metadata: {name}:{key}")
        start, end = offsets
        expected_bytes = DTYPE_BYTES[dtype]
        for dim in shape:
            expected_bytes *= dim
        if start < 0 or end <= start or end - start != expected_bytes:
            raise RuntimeError(f"Invalid tensor byte extent: {name}:{key}")
        spans.append((start, end))
    previous = 0
    for start, end in sorted(spans):
        if start != previous:
            raise RuntimeError(f"Noncontiguous or overlapping tensor data: {name}")
        previous = end
    if previous != file_size - 8 - size:
        raise RuntimeError(f"Truncated or trailing tensor data: {name}")
    return data


def safetensors_header(path: Path) -> dict[str, dict]:
    """Bounded local-file header read, retained for offline unit tests."""
    file_size = path.stat().st_size
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Invalid checkpoint object {path.name}")
    with path.open("rb") as stream:
        return _parse_safetensors_header(stream, file_size, path.name)


def remote_header(fs, name: str, file_size: int) -> dict[str, dict]:
    """Use the supported HF bucket streaming API, not a fragile NFS handle.

    block_size=0 means HfFileSystemStreamFile reads only requested bytes
    without buffering the full potentially multi-GB shard.
    """
    uri = f"hf://buckets/{BUCKET}/IQ/runs/{RUN}/{name}"
    try:
        with fs.open(uri, "rb", block_size=0, cache_type="none") as stream:
            return _parse_safetensors_header(stream, file_size, name)
    except Exception as exc:
        raise RuntimeError(f"HF direct header read failed for {name}: {exc}") from exc


def checkpoint_json(fs, root: Path, name: str, *, read_mode: str) -> dict:
    """Read small metadata from the chosen source, with bounded HF fallback."""
    if read_mode != "hub":
        try:
            return json.loads((root / name).read_text(encoding="utf-8"))
        except OSError as exc:
            if read_mode != "auto" or exc.errno not in (errno.EIO, errno.ESTALE):
                raise
            print(f"IQ_METADATA_NFS_FALLBACK_JSON={name}", flush=True)
    uri = f"hf://buckets/{BUCKET}/IQ/runs/{RUN}/{name}"
    try:
        with fs.open(uri, "rb", block_size=0, cache_type="none") as stream:
            payload = stream.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            raise RuntimeError(f"Oversize checkpoint metadata: {name}")
        return json.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"Cannot directly read HF checkpoint metadata {name}: {exc}") from exc


def expected_names() -> set[str]:
    from scripts.glm53_full78_gate import expected_stage_shards
    names = {"model-global.safetensors", "model-native.safetensors"}
    for stage in range(78):
        names.update(n for n in expected_stage_shards(stage) if n.endswith(".safetensors"))
    names.update(f"model-mamba-{i:03d}.safetensors" for i in range(79))
    if len(names) != 20283:
        raise RuntimeError(f"Unexpected number of weight shards: {len(names)}")
    return names


def validate_bucket(root: Path, *, api_sizes: dict[str, int],
                    schedule: list[str], stages: list[dict], latent: int,
                    intermediate: int, fs=None, read_mode: str = "mount") -> dict[str, str]:
    expected = expected_names()
    if set(api_sizes) != expected:
        raise RuntimeError(f"Incorrect remote weight inventory: missing={sorted(expected-set(api_sizes))[:5]}")
    index: dict[str, str] = {}
    for j, name in enumerate(sorted(expected), 1):
        path = root / name
        if read_mode == "hub":
            if fs is None:
                raise RuntimeError("HF filesystem client required for direct reads")
            entries = remote_header(fs, name, api_sizes[name])
        else:
            try:
                if not path.is_file() or path.stat().st_size != api_sizes[name]:
                    raise RuntimeError(f"Remote API vs NFS metadata mismatch: {name}")
                entries = safetensors_header(path)
            except OSError as exc:
                if read_mode != "auto" or fs is None or exc.errno not in (
                    errno.EIO, errno.ESTALE
                ):
                    raise RuntimeError(
                        f"NFS safetensors read failed for {name}: {exc}"
                    ) from exc
                print(f"IQ_METADATA_NFS_FALLBACK_HEADER={name}", flush=True)
                entries = remote_header(fs, name, api_sizes[name])
        if name == "model-global.safetensors":
            if set(entries) != {"embed_tokens.weight", "norm.weight", "lm_head.weight"}:
                raise RuntimeError("Invalid lexical foundation tensor keys")
        elif name == "model-native.safetensors":
            if any(not key.startswith(("attnres.", "reasoning.", "reasoning_injector.")) for key in entries):
                raise RuntimeError("Unexpected tensor in native-controller shard")
            if not all(any(key.startswith(prefix) for key in entries)
                       for prefix in ("attnres.", "reasoning.", "reasoning_injector.")):
                raise RuntimeError("Incomplete native-controller inventory")
        elif name.startswith("model-mamba-"):
            ordinal = int(name[12:15])
            physical = [i for i, kind in enumerate(schedule) if kind == "mamba3"][ordinal]
            prefix = f"layers.{physical}."
            suffixes = {
                "norm.weight",
                "mamba.core.in_proj.weight", "mamba.core.dt_bias",
                "mamba.core.B_bias", "mamba.core.C_bias",
                "mamba.core.B_norm.weight", "mamba.core.C_norm.weight",
                "mamba.core.mimo_x", "mamba.core.mimo_z",
                "mamba.core.mimo_o", "mamba.core.D",
                "mamba.core.out_proj.weight",
            }
            if set(entries) != {prefix + suffix for suffix in suffixes}:
                raise RuntimeError(f"Invalid Mamba physical placement/tensor inventory: {name}")
        elif name.startswith("model-context-"):
            stage = stages[int(name[14:17])]
            physical = int(stage["context_physical_layer"])
            mode = schedule[physical]
            base = f"layers.{physical}"
            attn = base + ".attention."
            common = {
                base + ".norm.weight",
                *(attn + suffix for suffix in (
                    "q_a_proj.weight", "q_a_norm.weight", "q_b_proj.weight",
                    "kv_proj.weight", "kv_norm.weight",
                    "compressor_gate_proj.weight", "compressor_position_bias",
                    "compressor_kv_norm.weight", "compressor_kv_proj.weight",
                    "sinks", "output.weight", "output.out_proj.weight",
                )),
            }
            if mode == "csa":
                common.update(attn + suffix for suffix in (
                    "index_kv_proj.weight", "index_kv_norm.weight",
                    "index_kv_norm.bias", "index_q_proj.weight",
                    "index_weight_proj.weight",
                ))
            if mode not in ("csa", "hca") or set(entries) != common:
                raise RuntimeError(f"Invalid context physical placement/tensor inventory: {name}")
        elif name.startswith("model-moe-"):
            stage = stages[int(name[10:13])]
            physical = int(stage["moe_physical_layer"])
            if schedule[physical] != "moe" or any(
                not k.startswith(f"layers.{physical}.") for k in entries
            ):
                raise RuntimeError(f"Invalid MoE physical placement: {name}")
            if "-expert-" in name:
                slot = int(name[-15:-12])
                prefix = f"layers.{physical}.moe.routed_experts.{slot}."
                expected_shapes = {
                    prefix + "gate_proj.weight": [intermediate, latent],
                    prefix + "up_proj.weight": [intermediate, latent],
                    prefix + "down_proj.weight": [latent, intermediate],
                }
                if set(entries) != set(expected_shapes):
                    raise RuntimeError(f"Wrong expert tensor inventory: {name}")
                for key, shape in expected_shapes.items():
                    if entries[key]["shape"] != shape or entries[key]["dtype"] != "BF16":
                        raise RuntimeError(f"Wrong expert layout/dtype: {name}:{key}")
            elif name.endswith("-common.safetensors"):
                prefix = f"layers.{physical}.moe."
                common = {
                    f"layers.{physical}.norm.weight",
                    prefix + "latent_norm.weight",
                    prefix + "latent_down.weight",
                    prefix + "latent_up.weight",
                    prefix + "router.weight",
                    prefix + "routing_bias",
                }
                if set(entries) != common:
                    raise RuntimeError(f"Wrong common MoE tensor inventory: {name}")
            elif name.endswith("-shared.safetensors"):
                prefix = f"layers.{physical}.moe.shared_experts.0."
                if set(entries) != {
                    prefix + "gate_proj.weight",
                    prefix + "up_proj.weight",
                    prefix + "down_proj.weight",
                }:
                    raise RuntimeError(f"Wrong shared MoE tensor inventory: {name}")
            else:
                raise RuntimeError(f"Unexpected MoE shard {name}")
        for key in entries:
            if key in index:
                raise RuntimeError(f"Duplicate tensor key {key} in {name} and {index[key]}")
            index[key] = name
        if j % 250 == 0 or j == len(expected):
            print(f"IQ_METADATA_HEADER_VERIFIED={j}/{len(expected)}", flush=True)
    return index



def build_fast_weight_map(
    schedule: list[str], stages: list[dict], native_keys: set[str]
) -> dict[str, str]:
    """Derive deterministic tensor-to-shard locations without opening each shard.

    This is NOT a full safetensors validation. Sampled headers validate this
    mapping for representative shard families; inference is still unproven.
    """
    if len(schedule) != 235 or len(stages) != 78:
        raise RuntimeError("Noncanonical fast-finalizer schedule/stage count")
    if not native_keys or not all(
        k.startswith(("attnres.", "reasoning.", "reasoning_injector."))
        for k in native_keys
    ) or not all(
        any(k.startswith(prefix) for k in native_keys)
        for prefix in ("attnres.", "reasoning.", "reasoning_injector.")
    ):
        raise RuntimeError("Invalid native-controller tensor keys")
    mapping: dict[str, str] = {}

    def add(name: str, keys) -> None:
        for key in keys:
            if key in mapping:
                raise RuntimeError(f"Duplicate expected tensor key {key}")
            mapping[key] = name

    add("model-global.safetensors",
        ("embed_tokens.weight", "norm.weight", "lm_head.weight"))
    add("model-native.safetensors", native_keys)

    mamba_ord = 0
    for physical, kind in enumerate(schedule):
        if kind != "mamba3":
            continue
        p = f"layers.{physical}."
        add(f"model-mamba-{mamba_ord:03d}.safetensors", (
            p + suffix for suffix in (
                "norm.weight", "mamba.core.in_proj.weight",
                "mamba.core.dt_bias", "mamba.core.B_bias",
                "mamba.core.C_bias", "mamba.core.B_norm.weight",
                "mamba.core.C_norm.weight", "mamba.core.mimo_x",
                "mamba.core.mimo_z", "mamba.core.mimo_o",
                "mamba.core.D", "mamba.core.out_proj.weight",
            )
        ))
        mamba_ord += 1
    if mamba_ord != 79:
        raise RuntimeError("Expected exactly 79 mapped Mamba blocks")

    for stage_id, record in enumerate(stages):
        if record.get("stage") != stage_id or record.get("source_layer") != stage_id:
            raise RuntimeError(f"Bad stage mapping {stage_id}")
        ctx = int(record["context_physical_layer"])
        moe = int(record["moe_physical_layer"])
        if schedule[ctx] not in ("csa", "hca") or schedule[moe] != "moe":
            raise RuntimeError(f"Bad physical stage placement {stage_id}")
        context_prefix = f"layers.{ctx}"
        a = context_prefix + ".attention."
        keys = [context_prefix + ".norm.weight"]
        keys.extend(a + suffix for suffix in (
            "q_a_proj.weight", "q_a_norm.weight", "q_b_proj.weight",
            "kv_proj.weight", "kv_norm.weight", "compressor_gate_proj.weight",
            "compressor_position_bias", "compressor_kv_norm.weight",
            "compressor_kv_proj.weight", "sinks", "output.weight",
            "output.out_proj.weight",
        ))
        if schedule[ctx] == "csa":
            keys.extend(a + suffix for suffix in (
                "index_kv_proj.weight", "index_kv_norm.weight",
                "index_kv_norm.bias", "index_q_proj.weight",
                "index_weight_proj.weight",
            ))
        add(f"model-context-{stage_id:03d}.safetensors", keys)
        p = f"layers.{moe}.moe."
        add(f"model-moe-{stage_id:03d}-common.safetensors", (
            f"layers.{moe}.norm.weight",
            p + "latent_norm.weight", p + "latent_down.weight",
            p + "latent_up.weight", p + "router.weight",
            p + "routing_bias",
        ))
        add(f"model-moe-{stage_id:03d}-shared.safetensors", (
            p + "shared_experts.0." + suffix
            for suffix in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        ))
        for slot in range(256):
            add(f"model-moe-{stage_id:03d}-expert-{slot:03d}.safetensors", (
                p + f"routed_experts.{slot}." + suffix
                for suffix in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
            ))
    if set(mapping.values()) != expected_names():
        missing = expected_names() - set(mapping.values())
        raise RuntimeError(f"Fast mapping omitted shards: {sorted(missing)[:10]}")
    return mapping


def validate_fast_samples(
    fs, api_sizes: dict[str, int], weight_map: dict[str, str],
    *, latent: int, intermediate: int
) -> list[str]:
    """Check a deliberate cross-section of all shard families from HF API.

    All weight shard sizes are independently inventoried; a sample is NOT
    equivalent to full header, SHA256 or numerical verification.
    """
    samples = [
        "model-global.safetensors",
        "model-native.safetensors",
        "model-mamba-000.safetensors",
        "model-mamba-018.safetensors",
        "model-mamba-078.safetensors",
        "model-context-000.safetensors",
        "model-context-077.safetensors",
        "model-moe-002-common.safetensors",
        "model-moe-003-common.safetensors",
        "model-moe-003-shared.safetensors",
        "model-moe-002-expert-000.safetensors",
        "model-moe-005-expert-019.safetensors",
        "model-moe-077-expert-255.safetensors",
    ]
    by_file: dict[str, set[str]] = {name: set() for name in samples}
    for key, file in weight_map.items():
        if file in by_file:
            by_file[file].add(key)
    for name in samples:
        if name not in api_sizes:
            raise RuntimeError(f"Sample missing from remote inventory: {name}")
        entries = remote_header(fs, name, api_sizes[name])
        if set(entries) != by_file[name]:
            missing = by_file[name] - set(entries)
            extra = set(entries) - by_file[name]
            raise RuntimeError(
                f"Sample tensor layout mismatch {name}: "
                f"missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}"
            )
        if "-expert-" in name:
            slot = int(name[-15:-12])
            prefix = next(iter(entries)).split(".routed_experts.")[0]
            for suffix, shape in (
                ("gate_proj.weight", [intermediate, latent]),
                ("up_proj.weight", [intermediate, latent]),
                ("down_proj.weight", [latent, intermediate]),
            ):
                key = f"{prefix}.routed_experts.{slot}.{suffix}"
                if entries[key]["shape"] != shape or entries[key]["dtype"] != "BF16":
                    raise RuntimeError(f"Sample expert geometry mismatch {name}:{key}")
        print(f"IQ_FAST_SAMPLE_HEADER_OK={name} tensors={len(entries)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true",
                        help="Upload index/manifest/completion after full dry-run checks")
    parser.add_argument("--fast", action="store_true",
                        help="Deterministic tensor mapping + remote inventory + sampled "
                             "headers; avoids 20,283 individual shard reads")
    parser.add_argument("--probe-headers", action="store_true",
                        help="Read only two remote safetensors headers to verify the "
                             "HF streaming backend without scanning the checkpoint")
    parser.add_argument("--read-mode", choices=("auto", "hub", "mount"),
                        default="auto",
                        help="auto: mounted reads + authenticated streaming HF fallback on EIO; "
                             "hub: bypass NFS entirely; mount: NFS only")
    parser.add_argument("--mount", type=Path,
                        default=Path.home() / "iq-hf-output")
    parser.add_argument("--root", type=Path,
                        default=Path.home() / "IQ-transfer-full78")
    args = parser.parse_args()
    if args.fast:
        # NFS is known to return EIO. Fast mode avoids mounted reads entirely.
        args.read_mode = "hub"
    output = args.mount / "IQ" / "runs" / RUN
    calibration = args.root / "calibration"
    if args.read_mode != "hub" and not output.is_dir():
        raise RuntimeError("Existing readonly checkpoint mount unavailable")
    # Do not silently complete a potentially concurrent compile.
    if os.environ.get("IQ_CONFIRM_NO_OTHER_WRITER") != "1":
        raise RuntimeError("Set IQ_CONFIRM_NO_OTHER_WRITER=1 after checking no compiler is running")
    from huggingface_hub import HfFileSystem
    # One client per finalization. HF direct reads are streamed and never
    # deserialize model weights. No new S3 credentials or GPU are required.
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    if args.probe_headers:
        from huggingface_hub import get_bucket_paths_info
        probe_names = ["model-context-000.safetensors", "model-mamba-018.safetensors"]
        prefix = f"IQ/runs/{RUN}"
        infos = {item.path.removeprefix(prefix + "/"): item.size for item in
                 get_bucket_paths_info(
                     BUCKET, [f"{prefix}/{n}" for n in probe_names]
                 )}
        if set(infos) != set(probe_names):
            raise RuntimeError(f"Missing remote probe shards: {sorted(set(probe_names)-set(infos))}")
        for name in probe_names:
            header = remote_header(fs, name, infos[name])
            print(f"IQ_HF_STREAMING_HEADER_PROBE_OK={name} tensors={len(header)}", flush=True)
        return
    config = checkpoint_json(fs, output, "iq_config.json", read_mode=args.read_mode)
    fingerprint = sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    run_provenance = checkpoint_json(
        fs, output, "run-provenance.json", read_mode=args.read_mode
    )
    receipt = calibration / "calibration_sha256.json"
    receipt_obj = json.loads(receipt.read_text())
    if receipt_obj.get("schema_version") != 1 or len(receipt_obj.get("files", {})) != 866:
        raise RuntimeError("Bad calibration receipt: expected 866 verified entries")
    if run_provenance != {
        "schema_version": 1,
        "calibration_receipt_sha256": sha256(receipt.read_bytes()),
        "teacher_revision": GLM_REV,
        "mamba_revision": MAMBA_REV,
        "target_fingerprint": fingerprint,
    }:
        raise RuntimeError("Pinned run provenance or config fingerprint mismatch")

    stages = json.loads((calibration / "glm53_calibration.json").read_text())["stages"]
    schedule = config["schedule"]["layers"]
    stable = config["stable_moe"]
    if (len(stages) != 78 or len(schedule) != 235 or
        schedule.count("mamba3") != 79 or
        schedule.count("moe") != 78 or
        schedule.count("csa") + schedule.count("hca") != 78 or
        stable["num_experts"] != 256):
        raise RuntimeError("Noncanonical 78-stage / 79-Mamba configuration")
    source_map = {}
    selected_ids = {}
    for i, stage in enumerate(stages):
        if stage.get("stage") != i or stage.get("source_layer") != i:
            raise RuntimeError(f"Noncanonical stage source mapping: {i}")
        physical_context = int(stage["context_physical_layer"])
        physical_moe = int(stage["moe_physical_layer"])
        if schedule[physical_context] not in ("csa", "hca") or schedule[physical_moe] != "moe":
            raise RuntimeError(f"Wrong physical schedule for stage {i}")
        source_map[str(i)] = i
        if i < 3:
            if stage.get("expert_usage") is not None or not stage.get("dense_intermediate_map"):
                raise RuntimeError(f"Bad dense stage calibration: {i}")
            selected_ids[str(i)] = None
        else:
            import numpy as np
            usage_file = stage.get("expert_usage")
            if not usage_file:
                raise RuntimeError(f"Missing expert usage at stage {i}")
            usage = np.load(calibration / usage_file, allow_pickle=False).reshape(-1)
            if len(usage) != 256 or not np.isfinite(usage).all() or (usage < 0).any() or float(usage.sum()) <= 0:
                raise RuntimeError(f"Bad expert usage data at stage {i}")
            selected_ids[str(i)] = [int(k) for k in np.lexsort(
                (np.arange(len(usage)), -usage)
            )]
        if not args.fast or i in (0, 39, 77):
            report = checkpoint_json(
                fs, output, f"model-moe-{i:03d}-projection.json",
                read_mode=args.read_mode,
            )
            if report.get("stage") != i or report.get("source_layer") != i:
                raise RuntimeError(f"Invalid saved stage projection receipt {i}")

    # Only reads official donor CONFIG and INDEX. Never reads any donor weight.
    from iq_transfer.glm53_job import validate_glm53_streaming_donor
    donor = validate_glm53_streaming_donor(
        checkpoint=args.root / "donors" / "glm53",
        checkpoint_revision=GLM_REV, donor_license="GLM-5.3",
    )
    print("IQ_FINALIZER_DONOR_METADATA_VERIFIED=1", flush=True)

    from huggingface_hub import get_bucket_paths_info
    names = sorted(expected_names())
    prefix = f"IQ/runs/{RUN}"
    extra = (
        [f"model-moe-{i:03d}-projection.json" for i in range(78)]
        if args.fast else []
    )
    remote_info = {
        item.path.removeprefix(prefix + "/"): item.size for item in
        get_bucket_paths_info(
            BUCKET, [f"{prefix}/{n}" for n in names + extra]
        )
    }
    missing = set(names + extra) - set(remote_info)
    invalid = [
        n for n, size in remote_info.items()
        if not isinstance(size, int) or size < (16 if n.endswith(".safetensors") else 1)
    ]
    if missing or invalid:
        raise RuntimeError(
            f"HF remote checkpoint incomplete: missing={sorted(missing)[:10]} "
            f"invalid={invalid[:10]}"
        )
    api = {name: remote_info[name] for name in names}
    samples = []
    if args.fast:
        native = remote_header(fs, "model-native.safetensors",
                               api["model-native.safetensors"])
        weight_map = build_fast_weight_map(schedule, stages, set(native))
        samples = validate_fast_samples(
            fs, api, weight_map, latent=int(stable["latent_size"]),
            intermediate=int(stable["expert_intermediate_size"]),
        )
        print(
            f"IQ_FAST_REMOTE_INVENTORY_OK=shards:{len(names)} "
            f"receipts:{len(extra)} samples:{len(samples)}",
            flush=True,
        )
    else:
        weight_map = validate_bucket(
            output, api_sizes=api, schedule=schedule, stages=stages,
            latent=int(stable["latent_size"]),
            intermediate=int(stable["expert_intermediate_size"]),
            fs=fs, read_mode=args.read_mode,
        )
    total_size = sum(api.values())
    provenance = {
        "schema_version": 2,
        "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
        "foundation": {
            "repo": "state-spaces/mamba3-mimo-1.5b",
            "revision": MAMBA_REV,
            "role": "pretrained lexical + recurrent foundation",
            "width_transfer": "2048->4096 exact replication embedding",
        },
        "capability_source": {
            "repo": "zai-org/GLM-5.3-BF16",
            "revision": GLM_REV,
            "fingerprint": donor.manifest.fingerprint,
            "checkpoint_hash": donor.manifest.checkpoint_hash,
            "role": "WARM-remapped context/DSA/MoE capability weights",
        },
        "recipient_config_fingerprint": fingerprint,
        "calibration_mode": json.loads((calibration / "glm53_calibration.json").read_text())["calibration_mode"],
        "routed_experts_per_sparse_stage": 256,
        "selected_source_expert_ids": selected_ids,
        "source_layer_map": source_map,
        "context": "MLA compressed-latent refactor + DSA direct-token indexer",
        "moe": "usage-selected expert subcloning into Stable LatentMoE",
        "mamba": "official Mamba-3 MIMO foundation; no GLM QKV-to-recurrence fabrication",
    }
    index = {"metadata": {"total_size": total_size},
             "weight_map": dict(sorted(weight_map.items()))}
    from huggingface_hub import get_bucket_paths_info as remote_path_info
    optional = {
        item.path.removeprefix(prefix + "/")
        for item in remote_path_info(BUCKET, [
            f"{prefix}/tokenizer.json",
            f"{prefix}/tokenizer_config.json",
        ])
    }
    complete = {
        "status": "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED",
        "source_layers": 78, "mapped_layers": 78,
        "routed_source_layers": 75, "routed_experts_per_layer": 256,
        "mapped_routed_expert_slots": 19200,
        "not_exact_bitwise_weight_reuse": True,
        "tokenizer_packaged": {
            "tokenizer.json", "tokenizer_config.json"
        }.issubset(optional),
        "inference_ready": False,
        "end_to_end_knowledge_retention_proven": False,
    }
    print(f"IQ_METADATA_FINALIZER_READY shards={len(names)} tensors={len(weight_map)} "
          f"bytes={total_size}", flush=True)
    if args.fast:
        complete["metadata_validation"] = (
            "remote_inventory_deterministic_map_"
            "and_13_sampled_headers_no_full_tensor_validation"
        )
        complete["metadata_sampled_shards"] = len(samples)
    if not args.publish:
        print(
            "IQ_FAST_METADATA_DRY_RUN_OK_NO_WRITES"
            if args.fast else "IQ_METADATA_FINALIZER_DRY_RUN_OK_NO_WRITES",
            flush=True,
        )
        return
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Set HF_TOKEN to publish via authorized Hub API")
    from iq_transfer.glm53_bucket_persistence import (
        write_checkpoint_text, remote_checkpoint_sizes,
    )
    os.environ.update({
        "IQ_HF_DURABLE_OUTPUT": str(output.resolve()),
        "IQ_HF_DURABLE_BUCKET": BUCKET,
        "IQ_HF_DURABLE_PREFIX": prefix,
        "IQ_HF_DURABLE_STAGING": str((args.root / "shard-staging").resolve()),
    })
    # Idempotent publishing verifies exact content through SHA256 readback.
    for filename, data in (
        ("model.safetensors.index.json", index),
        ("transplant_manifest.json", provenance),
    ):
        write_checkpoint_text(output, filename, json.dumps(
            data, indent=2, sort_keys=True
        ) + "\n")
    remote_checkpoint_sizes(output, names + [
        "model.safetensors.index.json", "transplant_manifest.json",
        "iq_config.json", "run-provenance.json"
    ])
    write_checkpoint_text(output, "COMPILE_COMPLETE.json", json.dumps(
        complete, sort_keys=True
    ) + "\n")
    remote_checkpoint_sizes(output, [
        "model.safetensors.index.json", "transplant_manifest.json",
        "COMPILE_COMPLETE.json"
    ])
    print("IQ_METADATA_FINALIZER_COMMITTED_INFERENCE_READY_FALSE", flush=True)


if __name__ == "__main__":
    main()
