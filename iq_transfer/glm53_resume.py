"""Resume completed GLM->IQ shards by reading safetensors headers, not tensor data."""
from __future__ import annotations

import re
import os
from pathlib import Path
from safetensors import safe_open


class GLM53ResumeError(RuntimeError):
    pass


_SHARD_RE = re.compile(
    r"model-(?:global|native|context-\d{3}|mamba-\d{3}"
    r"|moe-\d{3}(?:-common|-shared|-expert-\d{3})?)\.safetensors"
)


def register_saved_shard(
    output: Path,
    filename: str,
    weight_map: dict[str, str],
    *,
    expected_keys: set[str] | None = None,
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    layer: int | None = None,
) -> bool:
    if not _SHARD_RE.fullmatch(filename):
        raise GLM53ResumeError(f"invalid checkpoint shard filename {filename}")
    path = output / filename
    if not path.exists():
        return False
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 16:
        raise GLM53ResumeError(f"invalid existing shard: {filename}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            keys = set(reader.keys())
            if not keys or (expected_keys is not None and keys != expected_keys):
                raise GLM53ResumeError(f"incorrect tensor keys: {filename}")
            if expected_shapes is not None and keys != set(expected_shapes):
                raise GLM53ResumeError(f"incorrect expected-shape inventory: {filename}")
            if expected_dtypes is not None and keys != set(expected_dtypes):
                raise GLM53ResumeError(f"incorrect expected-dtype inventory: {filename}")
            for key in keys:
                slice_ = reader.get_slice(key)
                shape = tuple(int(n) for n in slice_.get_shape())
                if not shape or any(int(n) < 1 for n in shape):
                    raise GLM53ResumeError(f"invalid shape for {key}")
                if expected_shapes is not None and shape != expected_shapes[key]:
                    raise GLM53ResumeError(
                        f"wrong resumed tensor shape for {key}: "
                        f"got {shape}, expected {expected_shapes[key]}"
                    )
                if expected_dtypes is not None:
                    # The safetensors slice exposes dtype metadata without
                    # loading the complete large checkpoint tensor.
                    actual_dtype = slice_.get_dtype()
                    if actual_dtype != expected_dtypes[key]:
                        raise GLM53ResumeError(
                            f"wrong resumed tensor dtype for {key}: "
                            f"got {actual_dtype}, expected {expected_dtypes[key]}"
                        )
                if layer is not None and not key.startswith(f"layers.{layer}."):
                    raise GLM53ResumeError(f"incorrect physical layer {key}")
                if key in weight_map:
                    raise GLM53ResumeError(f"duplicate checkpoint tensor {key}")
    except Exception as exc:
        if isinstance(exc, GLM53ResumeError):
            raise
        raise GLM53ResumeError(f"corrupt existing safetensors: {filename}") from exc
    for key in keys:
        weight_map[key] = filename
    return True


def expected_expert_keys(physical: int, slot: int) -> set[str]:
    root = f"layers.{physical}.moe.routed_experts.{slot}"
    return {f"{root}.{name}_proj.weight" for name in ("gate", "up", "down")}


def expected_expert_shapes(
    physical: int, slot: int, *, latent_width: int, intermediate_width: int,
) -> dict[str, tuple[int, int]]:
    """Canonical GLM-to-IQ routed expert dimensions, excluding flat invalid slots."""
    if latent_width <= 0 or intermediate_width <= 0:
        raise GLM53ResumeError("positive expert dimensions required")
    root = f"layers.{physical}.moe.routed_experts.{slot}"
    return {
        f"{root}.gate_proj.weight": (intermediate_width, latent_width),
        f"{root}.up_proj.weight": (intermediate_width, latent_width),
        f"{root}.down_proj.weight": (latent_width, intermediate_width),
    }


def validate_legacy_partial(output: Path) -> None:
    """Explicitly adopt a stopped legacy run. Never use concurrently with writer."""
    if not output.is_dir():
        raise GLM53ResumeError("missing previous checkpoint output directory")
    if (output / "COMPILE_COMPLETE.json").exists():
        raise GLM53ResumeError("cannot resume a completed checkpoint")
    known = {"model.safetensors.index.json", "transplant_manifest.json",
             "iq_config.json", "GLM53_RESUME_MANIFEST.json", "INCOMPLETE", "run-provenance.json"}
    found = 0
    for path in output.iterdir():
        name = path.name
        if name in known or (name.startswith("model-moe-") and
                             name.endswith("-projection.json")):
            continue
        # Previous hf-mount advanced-write crashes can leave a committed
        # .model-...<random>.partial in the bucket. An orphan is NEVER
        # recognized as a complete shard; the missing FINAL file is rebuilt
        # and uploaded through the durable Hub API.
        if (
            os.environ.get("IQ_HF_DURABLE_OUTPUT")
            and re.fullmatch(
                r"\.model-(?:global|native|context-\d{3}|mamba-\d{3}|"
                r"moe-\d{3}(?:-common|-shared|-expert-\d{3})?)"
                r"\.safetensors\.[a-zA-Z0-9_]{8}\.partial", name
            )
            and path.is_file() and not path.is_symlink()
        ):
            continue
        if path.is_symlink() or not path.is_file() or not _SHARD_RE.fullmatch(name):
            raise GLM53ResumeError(f"unexpected partial checkpoint file: {name}")
        found += 1
    if found == 0:
        raise GLM53ResumeError("no partial shards to resume")
