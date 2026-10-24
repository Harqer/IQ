"""Export original selected GLM-5.3 MoE tensors with BF16 bits unchanged.

Donor-only intermediate: deliberately NOT an IQ checkpoint. No attention
matrices are fetched or transformed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from .complete_transplant import canonical_glm53_config
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class RawExpertExportError(RuntimeError):
    pass


def _int(x: Any) -> bool:
    return type(x) is int


def expert_keys(prefix: str) -> tuple[str, str, str]:
    return tuple(f"{prefix}.{part}_proj.weight" for part in ("gate", "up", "down"))


def validate_export_plan(
    plan: Mapping[str, Any],
    weight_map: Mapping[str, str],
    *,
    first_dense: int = 3,
    source_expert_count: int = 256,
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Verify a pinned official selective plan, before requesting any weights."""
    capacity = plan.get("target_experts")
    if not _int(capacity) or capacity not in (64, 128, 256):
        raise RawExpertExportError("expected 64/128/256 routed experts")
    if plan.get("schema_version") != 1 or plan.get("status") != "selective_metadata_plan_only":
        raise RawExpertExportError("unknown selective plan format")
    if plan.get("source_revision") != REVISION:
        raise RawExpertExportError("source revision is not pinned to official GLM")
    if plan.get("target_config_fingerprint") != canonical_glm53_config(
        routed_experts=capacity
    ).fingerprint:
        raise RawExpertExportError("target config fingerprint mismatch")
    if plan.get("official_total_shards") != 282:
        raise RawExpertExportError("source is not the official 282-shard GLM")
    layers = plan.get("source_layers")
    selected = plan.get("selected_donor_expert_ids")
    if (
        not isinstance(layers, list)
        or not isinstance(selected, list)
        or len(layers) != 24
        or len(selected) != 24
        or any(not _int(n) or not 0 <= n < 78 for n in layers)
        or layers != sorted(set(layers))
    ):
        raise RawExpertExportError("invalid 24-stage GLM layer mapping")
    allowed = plan.get("official_shards")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(s, str) for s in allowed)
        or len(allowed) != len(set(allowed))
    ):
        raise RawExpertExportError("invalid planned donor shards")
    shards = set(allowed)
    tasks: list[tuple[int, int, tuple[int, ...]]] = []
    for stage, (layer, ids) in enumerate(zip(layers, selected, strict=True)):
        if layer < first_dense:
            if ids is not None:
                raise RawExpertExportError("dense stage cannot contain routed IDs")
            continue
        if (
            not isinstance(ids, list)
            or len(ids) != capacity
            or any(not _int(i) or not 0 <= i < source_expert_count for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise RawExpertExportError(f"invalid selected donor expert IDs: stage {stage}")
        prefix = f"model.layers.{layer}.mlp"
        keys = [
            f"{prefix}.gate.weight",
            f"{prefix}.gate.e_score_correction_bias",
            *expert_keys(f"{prefix}.shared_experts"),
        ]
        for donor_id in ids:
            keys.extend(expert_keys(f"{prefix}.experts.{donor_id}"))
        for key in keys:
            if key not in weight_map:
                raise RawExpertExportError(f"official tensor index missing: {key}")
            if weight_map[key] not in shards:
                raise RawExpertExportError(f"tensor shard missing from planned shards: {key}")
        tasks.append((stage, layer, tuple(ids)))
    if len(tasks) != 23:
        raise RawExpertExportError("expected 23 sparse routed stages")
    return tasks


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _read(
    source: Any, name: str, shape: tuple[int, ...], *,
    allow_fp32: bool = False,
) -> torch.Tensor:
    if tuple(source.shape(name)) != shape:
        raise RawExpertExportError(f"unexpected GLM tensor shape for {name}")
    tensor = source.get(name)
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype not in (
            (torch.bfloat16, torch.float32) if allow_fp32
            else (torch.bfloat16,)
        )
        or tuple(tensor.shape) != shape
    ):
        raise RawExpertExportError(f"donor tensor has invalid shape, device or dtype: {name}")
    return tensor.detach().contiguous()


def _save(
    root: Path,
    filename: str,
    tensors: dict[str, torch.Tensor],
    mapping: dict[str, str],
    file_hashes: dict[str, str],
    tensor_hashes: dict[str, str],
    dtypes: dict[str, str],
) -> None:
    path = root / filename
    if path.exists() or any(key in mapping for key in tensors):
        raise RawExpertExportError(f"duplicate output file or tensor: {filename}")
    sha = {key: _sha_tensor(value) for key, value in tensors.items()}
    save_file(tensors, str(path))
    roundtrip = load_file(str(path), device="cpu")
    if set(roundtrip) != set(tensors) or any(
        t.dtype != tensors[k].dtype or _sha_tensor(t) != sha[k]
        for k, t in roundtrip.items()
    ):
        raise RawExpertExportError(f"BF16 roundtrip integrity failure: {filename}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    file_hashes[filename] = digest.hexdigest()
    for key, value in sha.items():
        tensor_hashes[key] = value
        dtypes[key] = str(tensors[key].dtype).removeprefix("torch.")
        mapping[key] = filename


def export_stage(
    *,
    source: Any,
    root: Path,
    stage: int,
    layer: int,
    ids: tuple[int, ...],
    hidden: int,
    intermediate: int,
    max_experts: int | None,
    mapping: dict[str, str],
    files: dict[str, str],
    tensors: dict[str, str],
    dtypes: dict[str, str],
) -> int:
    """Stream one donor expert at a time; include router and shared expert."""
    if max_experts is not None and (not _int(max_experts) or max_experts < 1):
        raise RawExpertExportError("max_experts must be positive")
    prefix = f"model.layers.{layer}.mlp"
    router = {
        f"{prefix}.gate.weight": _read(
            source, f"{prefix}.gate.weight", (256, hidden)
        ),
        f"{prefix}.gate.e_score_correction_bias": _read(
            source, f"{prefix}.gate.e_score_correction_bias", (256,),
            allow_fp32=True,
        ),
    }
    _save(root, f"stage-{stage:02d}-router.safetensors", router,
          mapping, files, tensors, dtypes)
    del router

    def write_expert(name: str, filename: str) -> None:
        shapes = (
            (intermediate, hidden), (intermediate, hidden),
            (hidden, intermediate),
        )
        group = {key: _read(source, key, shape)
                 for key, shape in zip(expert_keys(name), shapes, strict=True)}
        _save(root, filename, group, mapping, files, tensors, dtypes)
        del group

    write_expert(f"{prefix}.shared_experts", f"stage-{stage:02d}-shared.safetensors")
    exported = ids if max_experts is None else ids[:max_experts]
    for slot, donor_id in enumerate(exported):
        write_expert(
            f"{prefix}.experts.{donor_id}",
            f"stage-{stage:02d}-slot-{slot:03d}-donor-{donor_id:03d}.safetensors",
        )
    return len(exported)


def export_raw_selected_experts(
    *,
    source: Any,
    plan: Mapping[str, Any],
    tasks: list[tuple[int, int, tuple[int, ...]]],
    root: Path,
    hidden: int,
    intermediate: int,
    index_sha256: str,
    max_stages: int | None = None,
    max_experts: int | None = None,
) -> dict[str, Any]:
    """Create only a new donor bundle, never a runnable IQ checkpoint."""
    if not tasks or min(hidden, intermediate) < 1:
        raise RawExpertExportError("invalid donor geometry or no selected experts")
    if max_stages is not None and (not _int(max_stages) or max_stages < 1):
        raise RawExpertExportError("max_stages must be positive")
    root = Path(root)
    if root.is_symlink() or (
        root.exists() and (not root.is_dir() or any(root.iterdir()))
    ):
        raise RawExpertExportError("export requires a NEW empty directory")
    root.mkdir(parents=True, exist_ok=True)
    incomplete = root / "INCOMPLETE"
    incomplete.write_text(
        "Donor BF16 bundle export in progress; NOT an IQ checkpoint.\n",
        encoding="utf-8",
    )
    mapping: dict[str, str] = {}
    files: dict[str, str] = {}
    tensors: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    entries = []
    active = tasks if max_stages is None else tasks[:max_stages]
    for stage, layer, ids in active:
        count = export_stage(
            source=source, root=root, stage=stage, layer=layer, ids=ids,
            hidden=hidden, intermediate=intermediate,
            max_experts=max_experts, mapping=mapping,
            files=files, tensors=tensors, dtypes=dtypes,
        )
        entries.append({
            "stage": stage, "source_layer": layer,
            "planned_donor_ids": list(ids),
            "exported_donor_ids": list(ids[:count]),
            "exported_count": count,
        })
    complete = len(active) == len(tasks) and all(
        entry["exported_count"] == len(entry["planned_donor_ids"])
        for entry in entries
    )
    report = {
        "schema_version": 1,
        "artifact_type": "glm53_exact_bf16_donor_experts",
        "status": "complete_raw_donor_bundle" if complete else "partial_smoke_only",
        "not_an_iq_checkpoint": True,
        "source_repo": REPO,
        "source_revision": REVISION,
        "official_index_sha256": index_sha256,
        "target_experts": plan["target_experts"],
        "target_config_fingerprint": plan["target_config_fingerprint"],
        "layers": entries,
        "weight_map": dict(sorted(mapping.items())),
        "file_sha256": dict(sorted(files.items())),
        "tensor_sha256": dict(sorted(tensors.items())),
        "tensor_dtype": dict(sorted(dtypes.items())),
        "tensor_count": len(mapping),
        "note": "Exact original tensor bytes (BF16 experts; bias dtype preserved); no attention or IQ transforms.",
    }
    (root / "raw_expert_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report
