from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json

import torch

from iq_model import HybridLayerType, Mamba3MIMOConfig

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
    shard: str


@dataclass(frozen=True)
class Mamba3DirectTransferResult:
    output_dir: Path
    donor_sha256: str
    target_fingerprint: str
    placements: tuple[Mamba3LayerPlacement, ...]
    identity_mamba_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class Mamba3OverlayApplyReport:
    copied_parameters: tuple[str, ...]
    identity_parameters: tuple[str, ...]


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



def expand_mamba3_foundation_globals(
    *,
    source_config: Mamba3DonorConfig,
    state: Mapping[str, torch.Tensor],
    target_hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Widen Mamba lexical/global tensors while preserving source logits."""
    if target_hidden_size % source_config.d_model != 0:
        raise Mamba3DirectTransferError(
            "target hidden size must be an integer multiple of the Mamba foundation width"
        )
    factor = target_hidden_size // source_config.d_model
    if factor <= 0:
        raise Mamba3DirectTransferError("invalid Mamba foundation widening factor")
    embedding = _require_tensor(
        state,
        "backbone.embedding.weight",
        (source_config.vocab_size, source_config.d_model),
    )
    final_norm = _require_tensor(
        state,
        "backbone.norm_f.weight",
        (source_config.d_model,),
    )
    lm_head = _require_tensor(
        state,
        "lm_head.weight",
        (source_config.vocab_size, source_config.d_model),
    )
    return {
        "embed_tokens.weight": embedding.repeat((1, factor)).contiguous(),
        "norm.weight": final_norm.repeat(factor).contiguous(),
        "lm_head.weight": (
            lm_head.repeat((1, factor)) / float(factor)
        ).contiguous(),
    }


def load_official_mamba3_foundation_globals(
    *,
    checkpoint: str | Path,
    checkpoint_revision: str = MAMBA3_MIMO_15B_REVISION,
    verify_checkpoint_hash: bool = True,
    target_hidden_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Load the lexical/global state for the Mamba-3-founded IQ checkpoint.

    The official 1.5B model uses d_model=2048. IQ widens that representation by
    exact replication x -> [x, x]. To preserve the source function on that
    embedded subspace:
      embedding:  E -> [E, E]
      final norm: gamma -> [gamma, gamma]
      LM head:    W -> [W/2, W/2]
    so [x,x] @ [W/2,W/2]^T == x @ W^T.

    GLM-5.3 never owns these lexical tensors in the canonical IQ build.
    """
    checkpoint_dir = Path(checkpoint)
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not config_path.exists() or not weights_path.exists():
        raise Mamba3DirectTransferError(
            "checkpoint directory must contain config.json and pytorch_model.bin"
        )
    if checkpoint_revision != MAMBA3_MIMO_15B_REVISION:
        raise Mamba3DirectTransferError(
            "foundation transfer is pinned to the official Mamba-3 MIMO 1.5B revision "
            f"{MAMBA3_MIMO_15B_REVISION}; got {checkpoint_revision}"
        )
    donor = Mamba3DonorConfig.from_json(config_path)
    validate_official_mamba3_mimo_15b_config(donor)
    donor_sha = _sha256_file(weights_path)
    if verify_checkpoint_hash and donor_sha != MAMBA3_MIMO_15B_BIN_SHA256:
        raise Mamba3DirectTransferError(
            "Mamba-3 foundation checkpoint hash does not match the pinned artifact: "
            f"got {donor_sha}, expected {MAMBA3_MIMO_15B_BIN_SHA256}"
        )
    state = _load_official_state_dict(weights_path)
    validate_official_mamba3_mimo_state(donor, state)
    return expand_mamba3_foundation_globals(
        source_config=donor,
        state=state,
        target_hidden_size=target_hidden_size,
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
    target_mamba_ordinal: int,
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
    if target_mamba_ordinal < 0:
        raise Mamba3DirectTransferError("target_mamba_ordinal must be non-negative")

    source_layout = source_config.layout
    factor = _replication_factor(source_layout, target_layout)
    p = f"backbone.layers.{source_layer}"
    q = f"mamba_layers.{target_mamba_ordinal}"

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
        f"{q}.core.in_proj.weight": target_in,
        f"{q}.core.dt_bias": replicate_heads("dt_bias"),
        f"{q}.core.B_bias": replicate_head_rank_state("B_bias"),
        f"{q}.core.C_bias": replicate_head_rank_state("C_bias"),
        f"{q}.core.B_norm.weight": b_norm,
        f"{q}.core.C_norm.weight": c_norm,
        f"{q}.core.mimo_x": replicate_head_rank_dim("mimo_x"),
        f"{q}.core.mimo_z": replicate_head_rank_dim("mimo_z"),
        f"{q}.core.mimo_o": replicate_head_rank_dim("mimo_o"),
        f"{q}.core.D": replicate_heads("D"),
        f"{q}.core.out_proj.weight": target_out,
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


def _mamba_target_fingerprint(config: Mamba3MIMOConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _target_layout(config: Mamba3MIMOConfig) -> Mamba3Layout:
    return Mamba3Layout(
        d_model=config.d_model,
        d_state=config.d_state,
        expand=config.expand,
        headdim=config.headdim,
        ngroups=1,
        rope_fraction=config.rope_fraction,
        is_mimo=True,
        mimo_rank=config.mimo_rank,
    )


def _overlay_key_to_recipient(
    key: str,
    mamba_positions: tuple[int, ...],
) -> str:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "mamba_layers":
        raise Mamba3DirectTransferError(
            f"invalid canonical Mamba overlay key: {key}"
        )
    try:
        ordinal = int(parts[1])
    except ValueError as exc:
        raise Mamba3DirectTransferError(
            f"invalid Mamba ordinal in overlay key: {key}"
        ) from exc
    if ordinal < 0 or ordinal >= len(mamba_positions):
        raise Mamba3DirectTransferError(
            f"overlay Mamba ordinal {ordinal} outside recipient Mamba depth"
        )
    physical = mamba_positions[ordinal]
    suffix = parts[2]
    if suffix == "norm.weight":
        return f"layers.{physical}.norm.weight"
    if suffix.startswith("core."):
        return f"layers.{physical}.mamba.{suffix}"
    raise Mamba3DirectTransferError(
        f"unsupported canonical Mamba overlay suffix: {suffix}"
    )


def apply_mamba3_transplant_overlay(
    model: torch.nn.Module,
    overlay_dir: str | Path,
) -> Mamba3OverlayApplyReport:
    """Apply a schedule-independent Mamba overlay to an IQ hybrid model.

    The compiled artifact is indexed by Mamba ordinal, not physical hybrid-layer
    position. At application time the model's explicit HybridSchedule resolves
    those ordinals to physical IQ layers. Donor-depth gaps become exact residual
    identities by zeroing only their Mamba output projection.
    """
    root = Path(overlay_dir)
    try:
        manifest = json.loads(
            (root / "mamba3_transplant.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Mamba3DirectTransferError(
            f"invalid Mamba-3 transplant manifest in {root}"
        ) from exc
    if manifest.get("artifact_type") != "iq_mamba3_direct_transplant":
        raise Mamba3DirectTransferError(
            "overlay artifact_type is not Mamba-3 direct transplant"
        )

    model_config = getattr(model, "config", None)
    if model_config is None:
        raise Mamba3DirectTransferError(
            "recipient model must expose IQHybridConfig as .config"
        )
    mamba_config = getattr(model_config, "mamba3", None)
    schedule = getattr(model_config, "schedule", None)
    if mamba_config is None or schedule is None:
        raise Mamba3DirectTransferError(
            "recipient model config must expose mamba3 and schedule"
        )

    target = manifest.get("target")
    if not isinstance(target, dict) or not isinstance(target.get("mamba3"), dict):
        raise Mamba3DirectTransferError("transplant manifest is missing target Mamba config")
    expected_target = Mamba3MIMOConfig(**target["mamba3"])
    if asdict(mamba_config) != asdict(expected_target):
        raise Mamba3DirectTransferError(
            "recipient Mamba-3 config does not match compiled transplant target"
        )
    model_rms_eps = float(getattr(getattr(model, "model_config", None), "rms_norm_eps", -1))
    if abs(model_rms_eps - float(target.get("rms_norm_eps", 1e-5))) > 1e-12:
        raise Mamba3DirectTransferError(
            "recipient RMSNorm epsilon does not match transplant target"
        )

    mamba_positions = tuple(schedule.positions(HybridLayerType.MAMBA3))
    if len(mamba_positions) != expected_target.num_layers:
        raise Mamba3DirectTransferError(
            "recipient schedule Mamba count does not match compiled target"
        )

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise Mamba3DirectTransferError(
            "safetensors is required to apply transplant artifacts"
        ) from exc

    parameters = dict(model.named_parameters())
    copied: list[str] = []
    with torch.no_grad():
        for placement in manifest.get("placements", []):
            if not isinstance(placement, dict) or "shard" not in placement:
                raise Mamba3DirectTransferError(
                    "invalid placement entry in transplant manifest"
                )
            shard = load_file(str(root / str(placement["shard"])), device="cpu")
            for overlay_key, value in shard.items():
                key = _overlay_key_to_recipient(overlay_key, mamba_positions)
                try:
                    recipient_parameter = parameters[key]
                except KeyError as exc:
                    raise Mamba3DirectTransferError(
                        f"recipient model is missing overlay parameter: {key}"
                    ) from exc
                if _shape(recipient_parameter) != _shape(value):
                    raise Mamba3DirectTransferError(
                        f"overlay shape mismatch for {key}: got {_shape(value)}, "
                        f"recipient expects {_shape(recipient_parameter)}"
                    )
                recipient_parameter.copy_(
                    value.to(
                        device=recipient_parameter.device,
                        dtype=recipient_parameter.dtype,
                    )
                )
                copied.append(key)

        identity_parameters: list[str] = []
        for ordinal in manifest.get("identity_mamba_ordinals", []):
            ordinal = int(ordinal)
            if ordinal < 0 or ordinal >= len(mamba_positions):
                raise Mamba3DirectTransferError(
                    f"identity Mamba ordinal {ordinal} outside recipient depth"
                )
            physical = mamba_positions[ordinal]
            key = f"layers.{physical}.mamba.core.out_proj.weight"
            try:
                recipient_parameter = parameters[key]
            except KeyError as exc:
                raise Mamba3DirectTransferError(
                    f"recipient model is missing identity Mamba parameter: {key}"
                ) from exc
            recipient_parameter.zero_()
            identity_parameters.append(key)

    return Mamba3OverlayApplyReport(
        copied_parameters=tuple(copied),
        identity_parameters=tuple(identity_parameters),
    )


def compile_official_mamba3_mimo_15b_transplant(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    checkpoint_revision: str = MAMBA3_MIMO_15B_REVISION,
    verify_checkpoint_hash: bool = True,
) -> Mamba3DirectTransferResult:
    """Compile the official 1.5B donor into IQ's frozen 4096x32 Mamba target.

    This stage intentionally does not depend on the hybrid schedule. It emits
    Mamba-ordinal shards, so the same compiled artifact can later be applied to
    any IQ schedule that contains the frozen 32 Mamba-3 MIMO slots.
    """
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

    target_config = Mamba3MIMOConfig.production_4096x32()
    target_layout = _target_layout(target_config)
    _replication_factor(donor.layout, target_layout)
    ordinals = evenly_spaced_layer_placements(
        donor.n_layer,
        target_config.num_layers,
    )

    state = _load_official_state_dict(weights_path)
    validate_official_mamba3_mimo_state(donor, state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Mamba-3 is the lexical foundation, not only a recurrent overlay.
    # Persist the exact function-preserving 2048->4096 lexical widening as part
    # of the same durable foundation artifact.
    foundation_globals = expand_mamba3_foundation_globals(
        source_config=donor,
        state=state,
        target_hidden_size=target_config.d_model,
    )
    _save_safetensors(
        output / "model-global.safetensors",
        {
            key: value.to(torch.bfloat16)
            for key, value in foundation_globals.items()
        },
    )
    del foundation_globals

    placements: list[Mamba3LayerPlacement] = []
    populated_ordinals = set(ordinals)
    for source_layer, target_ordinal in enumerate(ordinals):
        shard = f"mamba/layer-{target_ordinal:03d}.safetensors"
        tensors = expand_mamba3_layer(
            source_config=donor,
            target_layout=target_layout,
            source_layer=source_layer,
            target_mamba_ordinal=target_ordinal,
            state=state,
        )
        _save_safetensors(output / shard, tensors)
        placements.append(
            Mamba3LayerPlacement(
                source_layer=source_layer,
                target_mamba_ordinal=target_ordinal,
                shard=shard,
            )
        )
        del tensors

    identity_ordinals = tuple(
        ordinal
        for ordinal in range(target_config.num_layers)
        if ordinal not in populated_ordinals
    )
    target_fingerprint = _mamba_target_fingerprint(target_config)

    manifest = {
        "schema_version": 3,
        "artifact_type": "iq_mamba3_foundation",
        "method": "exact_replication_embedding_plus_identity_depth_expansion",
        "teacher_student_distillation": False,
        "donor": {
            "repo_id": MAMBA3_MIMO_15B_REPO,
            "revision": checkpoint_revision,
            "checkpoint_sha256": donor_sha,
            "config": asdict(donor),
        },
        "target": {
            "fingerprint": target_fingerprint,
            "mamba3": asdict(target_config),
            "rms_norm_eps": 1e-5,
            "schedule_independent": True,
        },
        "placements": [asdict(item) for item in placements],
        "identity_mamba_ordinals": list(identity_ordinals),
        "identity_rule": "zero_mamba_out_proj",
        "scope": {
            "transferred": [
                "embedding",
                "final_norm",
                "lm_head",
                "mamba3_mixer",
                "mamba_pre_norm"
            ],
            "global_shard": "model-global.safetensors",
            "not_transferred": [
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
        target_fingerprint=target_fingerprint,
        placements=tuple(placements),
        identity_mamba_ordinals=identity_ordinals,
    )
