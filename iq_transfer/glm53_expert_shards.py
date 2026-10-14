"""GLM-5.3 → IQ exact-expert-ID distributed safetensors conversion.

All 256 GLM experts are addressed, partitioned into disjoint contiguous EP
shards. Donor tensors are read lazily from real BF16 safetensors files. Learned
6144→4096/2048 coordinate maps are required and never synthesized. This
compiler handles dense/FFN, routing and expert weights, NOT MLA/DSA or Mamba.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import SafetensorsSource
from .glm53_calibration import GLM53CalibrationSolution
from .glm53_parallel_contract import GLM53Topology
from .complete_transplant import nvidia_parallel_78_config
from .glm53_donor_preflight import audit_glm53_donor, GLM53DonorPreflightError


class GLM53ExpertShardError(RuntimeError):
    pass


def expert_range(ep_rank: int, ep_size: int, *, total: int = 256) -> range:
    if (
        type(ep_rank) is not int or type(ep_size) is not int
        or ep_size < 1 or ep_rank < 0 or ep_rank >= ep_size
        or total % ep_size
    ):
        raise GLM53ExpertShardError("EP rank/size must partition every donor expert")
    per_rank = total // ep_size
    return range(ep_rank * per_rank, (ep_rank + 1) * per_rank)


def _projection_pair(
    matrix: Any, *, device: torch.device, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-row → target-row basis: Wt = Pout.T @ Ws @ pinv(Pin).T."""
    map_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    if map_tensor.ndim != 2 or ridge <= 0 or not bool(torch.isfinite(map_tensor).all()):
        raise GLM53ExpertShardError("invalid learned projection basis")
    # Regularized least squares avoids repeatedly computing an SVD for all
    # 75×256 experts. The ridge is inherited from genuine calibration.
    gram = map_tensor.T @ map_tensor
    eye = torch.eye(gram.size(0), dtype=gram.dtype, device=device)
    inverse = torch.linalg.solve(gram + ridge * eye, map_tensor.T)
    return map_tensor, inverse.T.contiguous()


def _linear(
    donor: torch.Tensor,
    *, left: torch.Tensor | None,
    right: torch.Tensor | None,
    expected: tuple[int, int],
) -> torch.Tensor:
    if donor.ndim != 2 or tuple(donor.shape) != expected:
        raise GLM53ExpertShardError(
            f"donor tensor geometry {tuple(donor.shape)} != expected {expected}"
        )
    if donor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise GLM53ExpertShardError(
            "FP8 cannot be interpreted directly; import/dequantize through GLM5Bridge"
        )
    result = donor.to(device=(
        left.device if left is not None else right.device
    ), dtype=torch.float32)
    if right is not None:
        result = result @ right
    if left is not None:
        result = left.T @ result
    if not bool(torch.isfinite(result).all()):
        raise GLM53ExpertShardError("nonfinite transformed GLM expert parameter")
    return result.to(dtype=torch.bfloat16, device="cpu").contiguous()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_glm53_expert_shard(
    *, donor_dir: str | Path, calibration_dir: str | Path,
    output_dir: str | Path, ep_rank: int, ep_size: int,
    donor_revision: str, device: str = "cpu",
    donor_identity: str = "zai-org/GLM-5.3-BF16",
) -> dict[str, Any]:
    """Convert only owned routed experts; rank zero owns replicated FFN/router.

    A manifest is written **last**; absent manifest means incomplete output.
    No conversion of unknown/missing tensor keys, fake experts, or FP8 bytes.
    """
    experts = expert_range(ep_rank, ep_size)
    donor_root = Path(donor_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise GLM53ExpertShardError("refuse to overwrite previous expert conversion")
    if not donor_revision or len(donor_revision) != 40 or any(
        c not in "0123456789abcdef" for c in donor_revision.lower()
    ):
        raise GLM53ExpertShardError("donor revision must be a pinned SHA-1 commit")
    torch_device = torch.device(device)
    if torch_device.type not in ("cpu", "cuda"):
        raise GLM53ExpertShardError("expert conversion requires CPU or CUDA")
    # Fail before creating *any* output or loading heavy weights. The
    # publisher's FP8 checkpoint is NOT a native BF16 source for this
    # converter; use its official BF16 variant or GLM5Bridge dequantization.
    inventory = audit_glm53_donor(
        donor_root, require_bf16=True, verify_headers=True,
        checkpoint_identity=donor_identity,
    )
    with (donor_root / "config.json").open(encoding="utf-8") as f:
        donor_config = json.load(f)
    topology = GLM53Topology.from_hf_config(donor_config)
    iq = nvidia_parallel_78_config(expert_parallel_size=ep_size)
    calibration = GLM53CalibrationSolution.load(calibration_dir)
    # Rank partitioning does not change learned weight coordinates.
    default_fingerprint = nvidia_parallel_78_config().fingerprint
    if calibration.target_config_fingerprint not in {
        iq.fingerprint, default_fingerprint
    }:
        raise GLM53ExpertShardError(
            "coordinate calibration does not target this IQ 78-block architecture"
        )
    if calibration.calibration_mode != "activation_refined":
        raise GLM53ExpertShardError(
            "full-expert transport requires measured activation-refined calibration"
        )
    if len(calibration.stages) != 78:
        raise GLM53ExpertShardError("expected calibrated maps for all 78 GLM stages")
    if any(stage.stage != i or stage.source_layer != i
           for i, stage in enumerate(calibration.stages)):
        raise GLM53ExpertShardError("full 78-block transfer requires exact stage alignment")
    source = SafetensorsSource(donor_root)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise GLM53ExpertShardError("safetensors is required") from exc

    destination.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "artifact_type": "iq_glm53_full_expert_ep_shard",
        "donor_repo": donor_identity,
        "donor_index_preflight": inventory.to_dict(),
        "donor_revision": donor_revision,
        "donor_config_sha256": _sha(donor_root / "config.json"),
        "donor_index_sha256": (
            _sha(donor_root / "model.safetensors.index.json")
            if (donor_root / "model.safetensors.index.json").is_file()
            else None
        ),
        "donor_revision_verified_against_hub": False,
        "target_fingerprint_ep": iq.fingerprint,
        "calibration_fingerprint": calibration.target_config_fingerprint,
        "ep_rank": ep_rank, "ep_size": ep_size,
        "global_expert_ids": list(experts),
        "all_256_expert_ids_preserved": True,
        "attention_converted": False,
        "mamba_converted": False,
        "verified_by_teacher": False,
        "files": {},
    }

    def write(name: str, tensors: Mapping[str, torch.Tensor]) -> None:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(dict(sorted(tensors.items())), str(path))
        manifest["files"][name] = {
            "sha256": _sha(path),
            "tensor_shapes": {k: list(t.shape) for k, t in tensors.items()},
        }

    for block, stage in enumerate(calibration.stages):
        base = f"model.layers.{block}.mlp"
        target = f"layers.{block}"
        # These source maps are measured, not guessed from tensor dimensions.
        residual, _ = _projection_pair(
            stage.residual_map.matrix, device=torch_device,
            ridge=stage.residual_map.ridge
        )
        input_basis, input_right = _projection_pair(
            stage.moe_input_map.matrix, device=torch_device,
            ridge=stage.moe_input_map.ridge
        )
        if block < 3:
            if ep_rank != 0:
                continue
            if stage.dense_intermediate_map is None:
                raise GLM53ExpertShardError("missing actual dense intermediate map")
            intermediate, intermediate_right = _projection_pair(
                stage.dense_intermediate_map.matrix, device=torch_device,
                ridge=stage.dense_intermediate_map.ridge
            )
            tensors = {
                f"{target}.dense_ffn.0.weight": torch.ones(
                    iq.model.hidden_size, dtype=torch.bfloat16
                ),
            }
            for kind in ("gate_proj", "up_proj"):
                key = f"{base}.{kind}.weight"
                tensors[f"{target}.dense_ffn.1.{kind}.weight"] = _linear(
                    source.get(key), left=intermediate,
                    right=input_right,
                    expected=(intermediate.size(0), input_basis.size(0)),
                )
            tensors[f"{target}.dense_ffn.1.down_proj.weight"] = _linear(
                source.get(f"{base}.down_proj.weight"),
                left=residual, right=intermediate_right,
                expected=(residual.size(0), intermediate.size(0)),
            )
            write(f"shared/layer-{block:03d}.safetensors", tensors)
            continue

        latent_basis, latent_right = _projection_pair(
            stage.latent_map.matrix, device=torch_device,
            ridge=stage.latent_map.ridge
        )
        if latent_basis.size(1) != iq.stable_moe.latent_size:
            raise GLM53ExpertShardError("learned latent map has wrong IQ width")
        if ep_rank == 0:
            router_key = f"{base}.gate.weight"
            bias_key = f"{base}.gate.e_score_correction_bias"
            router = _linear(
                source.get(router_key), left=None, right=input_right,
                expected=(256, input_basis.size(0)),
            )
            bias = source.get(bias_key).to(torch.float32)
            if tuple(bias.shape) != (256,) or not bool(torch.isfinite(bias).all()):
                raise GLM53ExpertShardError("GLM router correction bias invalid")
            from .glm53_moe import latent_codec_weights
            codec_down, codec_up = latent_codec_weights(
                stage.moe_input_map, stage.latent_map, stage.residual_map
            )
            shared = {
                f"{target}.moe.moe.router.weight": router,
                f"{target}.moe.moe.routing_bias": (bias - bias.mean()).contiguous(),
                f"{target}.moe.moe.latent_down.weight": torch.tensor(
                    codec_down, dtype=torch.bfloat16
                ),
                f"{target}.moe.moe.latent_up.weight": torch.tensor(
                    codec_up, dtype=torch.bfloat16
                ),
                f"{target}.moe.moe.latent_norm.weight": torch.ones(
                    iq.stable_moe.latent_size, dtype=torch.bfloat16
                ),
                f"{target}.moe.norm.weight": torch.ones(
                    iq.model.hidden_size, dtype=torch.bfloat16
                ),
            }
            for kind in ("gate_proj", "up_proj"):
                shared[f"{target}.moe.moe.shared_experts.0.{kind}.weight"] = _linear(
                    source.get(f"{base}.shared_experts.{kind}.weight"),
                    left=None, right=input_right,
                    expected=(iq.stable_moe.expert_intermediate_size, input_basis.size(0)),
                )
            shared[f"{target}.moe.moe.shared_experts.0.down_proj.weight"] = _linear(
                source.get(f"{base}.shared_experts.down_proj.weight"),
                left=residual, right=None,
                expected=(residual.size(0), iq.stable_moe.expert_intermediate_size),
            )
            write(f"shared/layer-{block:03d}.safetensors", shared)

        # Every global expert ID is retained; ownership changes only its
        # rank-local parameter key. No selection, merging or replication.
        for global_id in experts:
            local_id = global_id - experts.start
            source_prefix = f"{base}.experts.{global_id}"
            dest_prefix = f"{target}.moe.moe.routed_experts.{local_id}"
            tensors = {}
            for kind in ("gate_proj", "up_proj"):
                tensors[f"{dest_prefix}.{kind}.weight"] = _linear(
                    source.get(f"{source_prefix}.{kind}.weight"),
                    left=None, right=latent_right,
                    expected=(iq.stable_moe.expert_intermediate_size, latent_basis.size(0)),
                )
            tensors[f"{dest_prefix}.down_proj.weight"] = _linear(
                source.get(f"{source_prefix}.down_proj.weight"),
                left=latent_basis, right=None,
                expected=(latent_basis.size(0), iq.stable_moe.expert_intermediate_size),
            )
            write(f"ep-{ep_rank:03d}/layer-{block:03d}/expert-{global_id:03d}.safetensors", tensors)

    # All local shards must exist, with no implied missing experts.
    expected_routed = 75 * len(experts)
    actual_routed = sum(
        name.startswith(f"ep-{ep_rank:03d}/")
        for name in manifest["files"]
    )
    if actual_routed != expected_routed:
        raise GLM53ExpertShardError(
            f"missing transferred expert shards: {actual_routed} != {expected_routed}"
        )
    manifest["local_expert_shards"] = expected_routed
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return manifest


def _read_complete_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GLM53ExpertShardError("missing or invalid complete EP shard manifest") from exc
    if manifest.get("artifact_type") != "iq_glm53_full_expert_ep_shard":
        raise GLM53ExpertShardError("not a completed IQ GLM expert shard conversion")
    if manifest.get("attention_converted") is not False or manifest.get("mamba_converted") is not False:
        raise GLM53ExpertShardError("incorrect donor converter claims")
    return manifest


def load_glm53_expert_shards(
    model: torch.nn.Module, *,
    local_dir: str | Path,
    shared_dir: str | Path,
    ep_rank: int,
) -> tuple[str, ...]:
    """Validate every shard name/hash/shape before writing ANY model tensor.

    Each rank loads its own routed expert files plus the rank-zero replicated
    router/codec/shared expert/dense FFN files. This is a partial GLM donor
    overlay, NOT a complete IQ checkpoint.
    """
    local = Path(local_dir)
    shared = Path(shared_dir)
    owned = _read_complete_manifest(local)
    replicated = _read_complete_manifest(shared)
    if owned["ep_rank"] != ep_rank or replicated["ep_rank"] != 0:
        raise GLM53ExpertShardError("EP shard rank mismatch")
    if owned["ep_size"] != replicated["ep_size"]:
        raise GLM53ExpertShardError("EP shard world sizes disagree")
    if owned["donor_revision"] != replicated["donor_revision"]:
        raise GLM53ExpertShardError("replicated parameters came from another donor revision")
    config = getattr(model, "config", None)
    if (
        config is None or config.expert_parallel_size != owned["ep_size"]
        or config.fingerprint != owned["target_fingerprint_ep"]
        or config.fingerprint != replicated["target_fingerprint_ep"]
    ):
        raise GLM53ExpertShardError("destination IQ model config/EP does not match shards")
    expected_global = list(expert_range(ep_rank, owned["ep_size"]))
    if owned["global_expert_ids"] != expected_global:
        raise GLM53ExpertShardError("EP shard has missing/duplicated global expert IDs")
    if owned["local_expert_shards"] != len(expected_global) * 75:
        raise GLM53ExpertShardError("incomplete routed expert shard manifest")
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    objects = {**parameters, **buffers}
    all_files: list[tuple[Path, dict[str, Any]]] = []
    for directory, manifest, kind in (
        (local, owned, "ep"),
        (shared, replicated, "shared"),
    ):
        files = manifest["files"]
        if not isinstance(files, dict):
            raise GLM53ExpertShardError("shard inventory must be a mapping")
        prefix = f"ep-{ep_rank:03d}/" if kind == "ep" else "shared/"
        filtered = sorted(
            (name, record) for name, record in files.items()
            if name.startswith(prefix)
        )
        for relative, record in filtered:
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise GLM53ExpertShardError("shard path escapes output directory")
            if not path.is_file() or _sha(path) != record["sha256"]:
                raise GLM53ExpertShardError(f"missing or tampered GLM expert shard: {relative}")
            for name, shape in record["tensor_shapes"].items():
                target = objects.get(name)
                if target is None or list(target.shape) != shape:
                    raise GLM53ExpertShardError(
                        f"destination parameter missing or wrong shape: {name}"
                    )
            all_files.append((path, record))
    # Validate the REAL safetensors index/shape/dtype of every shard before
    # copying any weight, not merely the manifest's claimed tensor shapes.
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise GLM53ExpertShardError("safetensors is required") from exc
    for path, record in all_files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(record["tensor_shapes"]):
                raise GLM53ExpertShardError(
                    f"tampered tensor key index in {path.name}"
                )
            for name, shape in record["tensor_shapes"].items():
                actual = handle.get_slice(name)
                if actual.get_shape() != shape:
                    raise GLM53ExpertShardError(
                        f"tampered tensor shape in {path.name}: {name}"
                    )
                expected_dtype = "F32" if name.endswith("routing_bias") else "BF16"
                if actual.get_dtype() != expected_dtype:
                    raise GLM53ExpertShardError(
                        f"unsupported transferred tensor dtype in {path.name}: {name}"
                    )

    loaded_names = set()
    for _, record in all_files:
        for name in record["tensor_shapes"]:
            if name in loaded_names:
                raise GLM53ExpertShardError(f"duplicated destination tensor: {name}")
            loaded_names.add(name)
    needed_local = {
        f"layers.{layer}.moe.moe.routed_experts.{local_index}.{projection}.weight"
        for layer in range(3, 78)
        for local_index in range(len(expected_global))
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if not needed_local.issubset(loaded_names):
        raise GLM53ExpertShardError("missing routed expert projection in EP manifest")
    with torch.no_grad():
        for path, record in all_files:
            tensors = load_file(str(path), device="cpu")
            if set(tensors) != set(record["tensor_shapes"]):
                raise GLM53ExpertShardError("safetensors contents differ from verified manifest")
            for name, source in tensors.items():
                target = objects[name]
                if source.dtype != torch.bfloat16 and not name.endswith("routing_bias"):
                    raise GLM53ExpertShardError(f"unexpected dtype for {name}")
                target.copy_(source.to(device=target.device, dtype=target.dtype))
    return tuple(sorted(loaded_names))


def verify_complete_expert_parallel_set(
    directories: list[str | Path],
) -> dict[str, Any]:
    """Prove rank manifests jointly cover all 256 experts, without duplicates.

    This is a cheap metadata gate; per-shard hashes are independently checked
    by load_glm53_expert_shards on each participating EP rank.
    """
    if not directories:
        raise GLM53ExpertShardError("no EP output directories supplied")
    manifests = [_read_complete_manifest(Path(folder)) for folder in directories]
    ep_size = manifests[0]["ep_size"]
    if len(manifests) != ep_size:
        raise GLM53ExpertShardError(
            "one completed expert-shard directory is required per EP rank"
        )
    revision = manifests[0]["donor_revision"]
    target = manifests[0]["target_fingerprint_ep"]
    calibration = manifests[0]["calibration_fingerprint"]
    config_sha256 = manifests[0].get("donor_config_sha256")
    index_sha256 = manifests[0].get("donor_index_sha256")
    all_ids: list[int] = []
    ranks: list[int] = []
    for manifest in manifests:
        if (manifest["ep_size"] != ep_size
            or manifest["donor_revision"] != revision
            or manifest["target_fingerprint_ep"] != target
            or manifest["calibration_fingerprint"] != calibration
            or manifest.get("donor_config_sha256") != config_sha256
            or manifest.get("donor_index_sha256") != index_sha256):
            raise GLM53ExpertShardError("EP shards disagree on source or target provenance")
        rank = manifest["ep_rank"]
        ranks.append(rank)
        owned = list(expert_range(rank, ep_size))
        if manifest["global_expert_ids"] != owned:
            raise GLM53ExpertShardError("EP manifest contains misplaced routed experts")
        expected = 75 * len(owned)
        if manifest.get("local_expert_shards") != expected:
            raise GLM53ExpertShardError("EP manifest has incomplete expert tensor count")
        files = manifest.get("files", {})
        if len([key for key in files if key.startswith(f"ep-{rank:03d}/")]) != expected:
            raise GLM53ExpertShardError("EP manifest routed file inventory is incomplete")
        all_ids.extend(owned)
    if sorted(ranks) != list(range(ep_size)):
        raise GLM53ExpertShardError("duplicate or absent EP ranks")
    if sorted(all_ids) != list(range(256)):
        raise GLM53ExpertShardError("full 256 expert IDs are not covered")
    rank_zero = next(m for m in manifests if m["ep_rank"] == 0)
    if len([key for key in rank_zero["files"] if key.startswith("shared/")]) != 78:
        raise GLM53ExpertShardError("rank zero must publish all 78 shared/dense FFN shards")
    return {
        "ep_size": ep_size,
        "all_256_experts_covered": True,
        "routed_expert_layer_shards": 75 * 256,
        "shared_and_dense_layer_shards": 78,
        "donor_revision": revision,
        "target_config_fingerprint": target,
        "calibration_fingerprint": calibration,
        "status": "metadata_complete_numerical_parity_pending",
    }
