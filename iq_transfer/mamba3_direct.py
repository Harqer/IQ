from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json

import torch

from iq_model import HybridLayerType, IQHybridConfig, validate_canonical_hybrid_backbone

from .mamba3_init import Mamba3Layout


MAMBA3_MIMO_15B_REPO = "state-spaces/mamba3-mimo-1.5b"
MAMBA3_MIMO_15B_REVISION = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
MAMBA3_MIMO_15B_BIN_SHA256 = "139c09e9728d1f10ac7f4354d2ae29aaca41851a954aedd8f6728c05aeb6c744"


class Mamba3DirectTransferError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mamba3DonorConfig:
    d_model: int
    d_intermediate: int
    n_layer: int
    vocab_size: int
    d_state: int
    expand: float
    headdim: int
    ngroups: int
    rope_fraction: float
    chunk_size: int
    is_mimo: bool
    mimo_rank: int
    is_outproj_norm: bool
    rms_norm: bool
    tie_embeddings: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Mamba3DonorConfig":
        ssm = data.get("ssm_cfg")
        if not isinstance(ssm, Mapping):
            raise Mamba3DirectTransferError("Mamba-3 donor config is missing ssm_cfg")
        if str(ssm.get("layer")) != "Mamba3":
            raise Mamba3DirectTransferError(
                f"expected Mamba3 donor layer, got {ssm.get('layer')!r}"
            )
        try:
            config = cls(
                d_model=int(data["d_model"]),
                d_intermediate=int(data["d_intermediate"]),
                n_layer=int(data["n_layer"]),
                vocab_size=int(data["vocab_size"]),
                d_state=int(ssm["d_state"]),
                expand=float(ssm["expand"]),
                headdim=int(ssm["headdim"]),
                ngroups=int(ssm["ngroups"]),
                rope_fraction=float(ssm["rope_fraction"]),
                chunk_size=int(ssm["chunk_size"]),
                is_mimo=bool(ssm["is_mimo"]),
                mimo_rank=int(ssm.get("mimo_rank", 1)),
                is_outproj_norm=bool(ssm.get("is_outproj_norm", False)),
                rms_norm=bool(data["rms_norm"]),
                tie_embeddings=bool(data["tie_embeddings"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Mamba3DirectTransferError(
                f"invalid Mamba-3 donor config: {exc}"
            ) from exc
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "Mamba3DonorConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Mamba3DirectTransferError(
                f"cannot read Mamba-3 donor config: {path}"
            ) from exc
        if not isinstance(data, dict):
            raise Mamba3DirectTransferError("Mamba-3 donor config must be a JSON object")
        return cls.from_mapping(data)

    def validate(self) -> None:
        for name in (
            "d_model",
            "d_intermediate",
            "n_layer",
            "vocab_size",
            "d_state",
            "headdim",
            "ngroups",
            "mimo_rank",
        ):
            if int(getattr(self, name)) <= 0:
                raise Mamba3DirectTransferError(f"{name} must be positive")
        if self.expand <= 0:
            raise Mamba3DirectTransferError("expand must be positive")
        if not self.is_mimo or self.mimo_rank < 2:
            raise Mamba3DirectTransferError("donor must use Mamba-3 MIMO")
        if not self.rms_norm:
            raise Mamba3DirectTransferError("official transfer expects RMSNorm")
        if self.is_outproj_norm:
            raise Mamba3DirectTransferError(
                "out-projection normalization is not supported by the current IQ mapping"
            )

    @property
    def layout(self) -> Mamba3Layout:
        return Mamba3Layout(
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            ngroups=self.ngroups,
            rope_fraction=self.rope_fraction,
            is_mimo=self.is_mimo,
            mimo_rank=self.mimo_rank,
        )

    @property
    def nheads(self) -> int:
        return self.layout.nheads


@dataclass(frozen=True)
class Mamba3LayerPlacement:
    source_layer: int
    target_mamba_ordinal: int
    target_physical_layer: int
    shard: str


@dataclass(frozen=True)
class Mamba3DirectTransferResult:
    output_dir: Path
    donor_sha256: str
    recipient_fingerprint: str
    placements: tuple[Mamba3LayerPlacement, ...]
    identity_physical_layers: tuple[int, ...]


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Mamba3DirectTransferError(
            f"failed to load donor PyTorch checkpoint: {path}"
        ) from exc
    if not isinstance(raw, dict) or not raw:
        raise Mamba3DirectTransferError("donor checkpoint is not a non-empty state_dict")
    result: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise Mamba3DirectTransferError(
                "donor checkpoint must contain only string->Tensor state_dict entries"
            )
        result[key] = value.detach().cpu()
    return result


def _shape(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _require_tensor(
    state: Mapping[str, torch.Tensor],
    key: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    try:
        value = state[key]
    except KeyError as exc:
        raise Mamba3DirectTransferError(f"missing donor tensor: {key}") from exc
    actual = _shape(value)
    if actual != shape:
        raise Mamba3DirectTransferError(
            f"donor tensor shape mismatch for {key}: got {actual}, expected {shape}"
        )
    return value


def validate_official_mamba3_mimo_state(
    config: Mamba3DonorConfig,
    state: Mapping[str, torch.Tensor],
) -> None:
    layout = config.layout
    expected_global = {
        "backbone.embedding.weight": (config.vocab_size, config.d_model),
        "backbone.norm_f.weight": (config.d_model,),
        "lm_head.weight": (config.vocab_size, config.d_model),
    }
    for key, shape in expected_global.items():
        _require_tensor(state, key, shape)

    for layer in range(config.n_layer):
        p = f"backbone.layers.{layer}"
        expected = {
            f"{p}.norm.weight": (config.d_model,),
            f"{p}.mixer.in_proj.weight": layout.in_proj_shape,
            f"{p}.mixer.dt_bias": (layout.nheads,),
            f"{p}.mixer.B_bias": (
                layout.nheads,
                layout.effective_mimo_rank,
                layout.d_state,
            ),
            f"{p}.mixer.C_bias": (
                layout.nheads,
                layout.effective_mimo_rank,
                layout.d_state,
            ),
            f"{p}.mixer.B_norm.weight": (layout.d_state,),
            f"{p}.mixer.C_norm.weight": (layout.d_state,),
            f"{p}.mixer.mimo_x": (
                layout.nheads,
                layout.effective_mimo_rank,
                layout.headdim,
            ),
            f"{p}.mixer.mimo_z": (
                layout.nheads,
                layout.effective_mimo_rank,
                layout.headdim,
            ),
            f"{p}.mixer.mimo_o": (
                layout.nheads,
                layout.effective_mimo_rank,
                layout.headdim,
            ),
            f"{p}.mixer.D": (layout.nheads,),
            f"{p}.mixer.out_proj.weight": layout.out_proj_shape,
            f"{p}.norm2.weight": (config.d_model,),
            f"{p}.mlp.fc1.weight": (2 * config.d_intermediate, config.d_model),
            f"{p}.mlp.fc2.weight": (config.d_model, config.d_intermediate),
        }
        for key, shape in expected.items():
            _require_tensor(state, key, shape)


def validate_official_mamba3_mimo_15b_config(config: Mamba3DonorConfig) -> None:
    expected = {
        "d_model": 2048,
        "d_intermediate": 3824,
        "n_layer": 24,
        "vocab_size": 128256,
        "d_state": 128,
        "expand": 2.0,
        "headdim": 64,
        "ngroups": 1,
        "rope_fraction": 0.5,
        "chunk_size": 16,
        "is_mimo": True,
        "mimo_rank": 4,
        "is_outproj_norm": False,
        "rms_norm": True,
        "tie_embeddings": True,
    }
    mismatches = [
        f"{name}={getattr(config, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(config, name) != value
    ]
    if mismatches:
        raise Mamba3DirectTransferError(
            "checkpoint is not the pinned official Mamba-3 MIMO 1.5B architecture: "
            + "; ".join(mismatches)
        )


def evenly_spaced_layer_placements(
    source_layers: int,
    target_layers: int,
) -> tuple[int, ...]:
    if source_layers <= 0 or target_layers <= 0:
        raise Mamba3DirectTransferError("layer counts must be positive")
    if source_layers > target_layers:
        raise Mamba3DirectTransferError(
            "direct depth expansion requires target Mamba depth >= donor depth"
        )
    if source_layers == 1:
        return (0,)
    positions = tuple(
        round(i * (target_layers - 1) / (source_layers - 1))
        for i in range(source_layers)
    )
    if len(set(positions)) != source_layers:
        raise Mamba3DirectTransferError(
            "even depth placement produced duplicate target layers"
        )
    return positions


def _replication_factor(source: Mamba3Layout, target: Mamba3Layout) -> int:
    if target.d_model < source.d_model:
        raise Mamba3DirectTransferError(
            "current direct Mamba-3 path supports width preservation/expansion, not compression"
        )
    fixed_fields = (
        "d_state",
        "headdim",
        "ngroups",
        "rope_fraction",
        "effective_mimo_rank",
    )
    mismatches = [
        name
        for name in fixed_fields
        if getattr(source, name) != getattr(target, name)
    ]
    if mismatches:
        raise Mamba3DirectTransferError(
            "Mamba-3 recurrent semantics differ for: " + ", ".join(mismatches)
        )
    if target.d_model % source.d_model != 0:
        raise Mamba3DirectTransferError(
            "target d_model must be an integer multiple of donor d_model for exact replication embedding"
        )
    factor = target.d_model // source.d_model
    if target.d_inner != source.d_inner * factor:
        raise Mamba3DirectTransferError(
            "target d_inner must scale by the same factor as d_model"
        )
    if target.nheads != source.nheads * factor:
        raise Mamba3DirectTransferError(
            "target head count must scale by the same factor as d_model"
        )
    return factor


def _replicate_input_columns(weight: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 0:
        raise Mamba3DirectTransferError("replication factor must be positive")
    return torch.cat([weight / factor for _ in range(factor)], dim=1)


def expand_mamba3_layer(
    *,
    source_config: Mamba3DonorConfig,
    target_layout: Mamba3Layout,
    source_layer: int,
    target_physical_layer: int,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Embed one pretrained Mamba-3 layer by exact representation replication.

    For a width factor r, the donor residual representation x is embedded as
    E(x)=[x,...,x] (r copies). RMSNorm is therefore invariant. Linear input
    operators are expanded as [W/r,...,W/r], head-local operators are
    duplicated, and the output projection is tiled as W/r. Consequently the
    target Mamba block maps E(x) to E(f(x)) on the embedded donor subspace,
    rather than merely zero-padding weights.
    """
    if source_layer < 0 or source_layer >= source_config.n_layer:
        raise Mamba3DirectTransferError("source_layer is outside donor depth")
    if target_physical_layer < 0:
        raise Mamba3DirectTransferError("target_physical_layer must be non-negative")

    source_layout = source_config.layout
    factor = _replication_factor(source_layout, target_layout)
    p = f"backbone.layers.{source_layer}"
    q = f"layers.{target_physical_layer}"

    source_in = _require_tensor(
        state,
        f"{p}.mixer.in_proj.weight",
        source_layout.in_proj_shape,
    )
    target_in = torch.empty(
        target_layout.in_proj_shape,
        dtype=source_in.dtype,
    )
    source_slices = source_layout.slices()
    target_slices = target_layout.slices()
    for name in ("z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angle"):
        s_slice = source_slices[name]
        t_slice = target_slices[name]
        source_rows = source_in[s_slice, :]
        source_row_count = s_slice.stop - s_slice.start
        target_row_count = t_slice.stop - t_slice.start
        if target_row_count % source_row_count != 0:
            raise Mamba3DirectTransferError(
                f"target Mamba-3 {name} rows are not an integer multiple of donor rows"
            )
        row_factor = target_row_count // source_row_count
        expanded = _replicate_input_columns(source_rows, factor)
        if row_factor > 1:
            expanded = expanded.repeat((row_factor, 1))
        if _shape(expanded) != (target_row_count, target_layout.d_model):
            raise Mamba3DirectTransferError(
                f"internal {name} expansion produced {_shape(expanded)}"
            )
        target_in[t_slice, :].copy_(expanded)

    source_out = _require_tensor(
        state,
        f"{p}.mixer.out_proj.weight",
        source_layout.out_proj_shape,
    )
    target_out = source_out.repeat((factor, factor)) / factor
    if _shape(target_out) != target_layout.out_proj_shape:
        raise Mamba3DirectTransferError(
            f"internal out_proj expansion produced {_shape(target_out)}"
        )

    source_norm = _require_tensor(
        state,
        f"{p}.norm.weight",
        (source_layout.d_model,),
    )
    target_norm = source_norm.repeat(factor)

    def replicate_heads(name: str) -> torch.Tensor:
        source = _require_tensor(
            state,
            f"{p}.mixer.{name}",
            (source_layout.nheads,),
        )
        return source.repeat(factor)

    def replicate_head_rank_state(name: str) -> torch.Tensor:
        shape = (
            source_layout.nheads,
            source_layout.effective_mimo_rank,
            source_layout.d_state,
        )
        source = _require_tensor(state, f"{p}.mixer.{name}", shape)
        return source.repeat((factor, 1, 1))

    def replicate_head_rank_dim(name: str) -> torch.Tensor:
        shape = (
            source_layout.nheads,
            source_layout.effective_mimo_rank,
            source_layout.headdim,
        )
        source = _require_tensor(state, f"{p}.mixer.{name}", shape)
        return source.repeat((factor, 1, 1))

    b_norm = _require_tensor(
        state,
        f"{p}.mixer.B_norm.weight",
        (source_layout.d_state,),
    ).clone()
    c_norm = _require_tensor(
        state,
        f"{p}.mixer.C_norm.weight",
        (source_layout.d_state,),
    ).clone()

    return {
        f"{q}.norm.weight": target_norm,
        f"{q}.mamba.core.in_proj.weight": target_in,
        f"{q}.mamba.core.dt_bias": replicate_heads("dt_bias"),
        f"{q}.mamba.core.B_bias": replicate_head_rank_state("B_bias"),
        f"{q}.mamba.core.C_bias": replicate_head_rank_state("C_bias"),
        f"{q}.mamba.core.B_norm.weight": b_norm,
        f"{q}.mamba.core.C_norm.weight": c_norm,
        f"{q}.mamba.core.mimo_x": replicate_head_rank_dim("mimo_x"),
        f"{q}.mamba.core.mimo_z": replicate_head_rank_dim("mimo_z"),
        f"{q}.mamba.core.mimo_o": replicate_head_rank_dim("mimo_o"),
        f"{q}.mamba.core.D": replicate_heads("D"),
        f"{q}.mamba.core.out_proj.weight": target_out,
    }


def _save_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise Mamba3DirectTransferError(
            "safetensors is required to write transplant artifacts"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))


def compile_official_mamba3_mimo_15b_transplant(
    *,
    checkpoint: str | Path,
    recipient_config_path: str | Path,
    output_dir: str | Path,
    checkpoint_revision: str = MAMBA3_MIMO_15B_REVISION,
    verify_checkpoint_hash: bool = True,
) -> Mamba3DirectTransferResult:
    checkpoint_dir = Path(checkpoint)
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not config_path.exists() or not weights_path.exists():
        raise Mamba3DirectTransferError(
            "checkpoint directory must contain config.json and pytorch_model.bin"
        )
    if checkpoint_revision != MAMBA3_MIMO_15B_REVISION:
        raise Mamba3DirectTransferError(
            "direct transfer is pinned to the official Mamba-3 MIMO 1.5B revision "
            f"{MAMBA3_MIMO_15B_REVISION}; got {checkpoint_revision}"
        )

    donor = Mamba3DonorConfig.from_json(config_path)
    validate_official_mamba3_mimo_15b_config(donor)

    donor_sha = _sha256_file(weights_path)
    if verify_checkpoint_hash and donor_sha != MAMBA3_MIMO_15B_BIN_SHA256:
        raise Mamba3DirectTransferError(
            "Mamba-3 donor checkpoint hash does not match the pinned official artifact: "
            f"got {donor_sha}, expected {MAMBA3_MIMO_15B_BIN_SHA256}"
        )

    recipient = validate_canonical_hybrid_backbone(
        IQHybridConfig.from_json(str(recipient_config_path))
    )
    if abs(float(recipient.model.rms_norm_eps) - 1e-5) > 1e-12:
        raise Mamba3DirectTransferError(
            "exact Mamba-3 replication embedding requires recipient RMSNorm eps=1e-5 "
            "to match the official donor runtime"
        )
    target_layout = Mamba3Layout(
        d_model=recipient.mamba3.d_model,
        d_state=recipient.mamba3.d_state,
        expand=recipient.mamba3.expand,
        headdim=recipient.mamba3.headdim,
        ngroups=1,
        rope_fraction=recipient.mamba3.rope_fraction,
        is_mimo=True,
        mimo_rank=recipient.mamba3.mimo_rank,
    )
    _replication_factor(donor.layout, target_layout)

    mamba_positions = recipient.schedule.positions(HybridLayerType.MAMBA3)
    if len(mamba_positions) != recipient.mamba3.num_layers:
        raise Mamba3DirectTransferError(
            "recipient Mamba schedule/config disagreement"
        )
    ordinals = evenly_spaced_layer_placements(
        donor.n_layer,
        len(mamba_positions),
    )

    state = _load_official_state_dict(weights_path)
    validate_official_mamba3_mimo_state(donor, state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    placements: list[Mamba3LayerPlacement] = []
    populated_ordinals = set(ordinals)
    for source_layer, target_ordinal in enumerate(ordinals):
        physical = mamba_positions[target_ordinal]
        shard = f"mamba/layer-{target_ordinal:03d}-physical-{physical:03d}.safetensors"
        tensors = expand_mamba3_layer(
            source_config=donor,
            target_layout=target_layout,
            source_layer=source_layer,
            target_physical_layer=physical,
            state=state,
        )
        _save_safetensors(output / shard, tensors)
        placements.append(
            Mamba3LayerPlacement(
                source_layer=source_layer,
                target_mamba_ordinal=target_ordinal,
                target_physical_layer=physical,
                shard=shard,
            )
        )
        del tensors

    identity_physical = tuple(
        physical
        for ordinal, physical in enumerate(mamba_positions)
        if ordinal not in populated_ordinals
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "iq_mamba3_direct_transplant",
        "method": "exact_replication_embedding_plus_identity_depth_expansion",
        "teacher_student_distillation": False,
        "donor": {
            "repo_id": MAMBA3_MIMO_15B_REPO,
            "revision": checkpoint_revision,
            "checkpoint_sha256": donor_sha,
            "config": asdict(donor),
        },
        "recipient": {
            "config_fingerprint": recipient.fingerprint,
            "hidden_size": recipient.model.hidden_size,
            "mamba3": asdict(recipient.mamba3),
            "schedule_fingerprint": recipient.schedule.fingerprint,
        },
        "placements": [asdict(item) for item in placements],
        "identity_physical_layers": list(identity_physical),
        "identity_rule": "zero_mamba_out_proj",
        "scope": {
            "transferred": ["mamba3_mixer", "mamba_pre_norm"],
            "not_transferred": [
                "embedding",
                "lm_head",
                "donor_gated_mlp",
                "stable_latent_moe",
                "csa_hca",
                "block_attnres",
                "reasoning_recurrence",
                "adaptive_halting",
                "energy_critic",
            ],
        },
    }
    (output / "mamba3_transplant.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Mamba3DirectTransferResult(
        output_dir=output,
        donor_sha256=donor_sha,
        recipient_fingerprint=recipient.fingerprint,
        placements=tuple(placements),
        identity_physical_layers=identity_physical,
    )
