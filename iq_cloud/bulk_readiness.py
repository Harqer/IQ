"""Read-only consolidated IQ235 readiness and six real shard-header checks.

Nothing here starts GPUs, reads model tensor payloads, changes the original
Hugging Face bucket or certifies pretrained model quality.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_metadata import (
    BUCKET, RUN, EXPECTED_INDEX_SHA256,
    check_pinned_metadata_sha256, IQ235MetadataError,
)
from .remote_shard_loader import verify_safetensors_header

SAMPLE_PREFIXES = (
    ("global", "model-global.safetensors"),
    ("native", "model-native.safetensors"),
    ("mamba", "model-mamba-"),
    ("context", "model-context-"),
    ("moe_common", "model-moe-000-common"),
    ("moe_expert", "model-moe-000-expert-"),
)


def choose_representative_shards(weight_map: Mapping[str, str]) -> dict[str, str]:
    """Choose six distinct source shard categories, failing on absent types."""
    if not isinstance(weight_map, dict) or not weight_map:
        raise IQ235MetadataError("source checkpoint index has no tensors")
    names = set(weight_map.values())
    selected: dict[str, str] = {}
    for label, prefix in SAMPLE_PREFIXES:
        options = sorted(
            s for s in names
            if isinstance(s, str) and s.startswith(prefix)
            and s.endswith(".safetensors")
        )
        if not options:
            raise IQ235MetadataError(f"checkpoint is missing {label} shard type")
        selected[label] = options[0]
    if len(set(selected.values())) != len(SAMPLE_PREFIXES):
        raise IQ235MetadataError("representative shard types overlap")
    return selected


def compare_representative_headers(
    filesystem: Any,
    base_uri: str,
    source_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Header-key, shape, dtype and offset audit; no payload bytes consumed."""
    weight_map = source_index.get("weight_map")
    selected = choose_representative_shards(weight_map)
    by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor_name, shard in weight_map.items():
        by_shard[shard].add(tensor_name)
    details: dict[str, dict[str, Any]] = {}
    for category, filename in selected.items():
        uri = base_uri.rstrip("/") + "/" + filename
        size = filesystem.info(uri).get("size")
        if type(size) is not int:
            raise IQ235MetadataError("remote source shard size unavailable")
        with filesystem.open(uri, "rb", block_size=4096) as stream:
            header = verify_safetensors_header(
                stream, file_size=size, expected_keys=by_shard[filename],
            )
            consumed = stream.tell()
        if consumed > size or consumed > 8 * 1024 * 1024 + 8:
            raise IQ235MetadataError("header-only inspection crossed size cap")
        details[category] = {
            "filename": filename,
            "indexed_tensors": len(by_shard[filename]),
            "verified_header_tensors": len(header),
            "dtype_set": sorted({entry["dtype"] for entry in header.values()}),
            "file_size_bytes": size,
            "header_bytes_read": consumed,
            "tensor_payload_bytes_read": 0,
        }
    return {
        "sample_status": "representative_safetensors_headers_verified_only",
        "representative_shard_count": len(details),
        "samples": details,
        "all_20283_shard_headers_verified": False,
        "all_source_payloads_verified": False,
        "full_model_schema_match_verified": False,
        "inference_ready": False,
    }


def static_release_blockers(root: Path, *, approved_digest_pin: str | None) -> dict[str, Any]:
    """Report evidence from repository source, not caller readiness booleans."""
    paths = {
        "runtime": root / "iq_cloud/full_iq235.py",
        "stage": root / "scripts/stage_iq235_baseten.py",
        "trainer": root / "iq_training/glm53_lexical_train.py",
        "marker": root / "configs/iq235_source_compile_status.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise IQ235MetadataError(f"required source file missing: {path.name}")
    runtime = paths["runtime"].read_text(encoding="utf-8")
    stage = paths["stage"].read_text(encoding="utf-8")
    trainer = paths["trainer"].read_text(encoding="utf-8")
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    if marker.get("inference_ready") is not False:
        raise IQ235MetadataError("untrusted compile marker claims inference readiness")
    return {
        "status": "production_inference_blocked",
        "source_index_sha256": EXPECTED_INDEX_SHA256,
        "source_compile_status": marker.get("status"),
        "source_compilation_inference_ready": False,
        "source_shard_manifest_pinned": approved_digest_pin is not None,
        "source_payload_sha256_verified": False,
        "all_source_model_weights_loaded": False,
        "actual_model_schema_matched_all_shards": False,
        "runtime_restricts_architecture_only": False,
        "default_deployment_is_architecture_only": (
            'IQ_PHASE: architecture_only' in
            (root / "deploy/baseten_full_iq/config.yaml").read_text(encoding="utf-8")
        ),
        "source_vocab_experimental_path_integrated": (
            'def load_pretrained_source(' in runtime
            and 'install_source_lexical=True' in runtime
        ),
        "source_vocab_verified_mode_blocked_on_manifest_pin": approved_digest_pin is None,
        "single_pass_exploratory_integrated": (
            '"pretrained_source_exploratory"' in runtime
            and "experimental_snapshot_path=" in runtime
        ),
        "single_pass_exploratory_executed": False,
        "exploratory_manifest_and_inference_result_reviewed": False,
        "runtime_injects_checkpoint_weights": "load_verified_full235_body(" in runtime,
        "baseten_stage_includes_loader": (
            '"full235_weight_loader.py"' in stage
            and '"remote_shard_loader.py"' in stage
        ),
        "lexical_production_training_blocked": (
            "verified full-checkpoint loader/attestation" in trainer
        ),
        "trained_glm_lexical_overlay_verified": False,
        "actual_full235_cuda_forward_verified": False,
        "heldout_generation_quality_verified": False,
        "readiness_to_collect_inference_api_key": False,
        "production_inference_ready": False,
        "steps_before_real_pretrained_inference": [
            "Confirm exact CUDA Mamba3 runtime, provider capacity, and a durable output mount",
            "Explicitly authorize billable H200x8 and streaming 539GB source weights",
            "Require opt-in experimental unreviewed provenance acceptance",
            "Stream/sha256/load each source shard exactly once while checking structural integrity",
            "Save unreviewed bulk manifest and separate inference outcome report",
            "Review traces and source Llama-ID generation before approving any digest pin",
        ],
        "steps_before_verified_production_inference": [
            "Independently review/checkpoint donor provenance and every source shard digest",
            "Pin the fully reviewed source manifest before the verified production load",
            "Re-run strict source/tensor validation on a supported CUDA runtime",
            "Validate full-model output quality and release readiness",
            "Treat GLM tokenizer remapping as a separate future experiment",
        ],
    }


def remote_header_audit(*, filesystem=None) -> dict[str, Any]:
    """Read pinned source JSON and six small remote SafeTensors headers only."""
    if filesystem is None:
        from huggingface_hub import HfFileSystem
        filesystem = HfFileSystem(block_size=4096)
    base = f"hf://buckets/{BUCKET}/{RUN}"
    def read_bounded(name: str, limit: int) -> bytes:
        with filesystem.open(base + "/" + name, "rb", block_size=4096) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise IQ235MetadataError(f"source metadata too large: {name}")
        return data
    config = read_bounded("iq_config.json", 100000)
    index = read_bounded("model.safetensors.index.json", 8000000)
    check_pinned_metadata_sha256(config, index)
    return compare_representative_headers(filesystem, base, json.loads(index))
