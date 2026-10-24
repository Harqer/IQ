"""Resume completed GLM->IQ shards by reading safetensors headers, not tensor data."""
from __future__ import annotations

import re
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
            for key in keys:
                shape = reader.get_slice(key).get_shape()
                if not shape or any(int(n) < 1 for n in shape):
                    raise GLM53ResumeError(f"invalid shape for {key}")
                if layer is not None and not key.startswith(f"layers.{layer}."):
                    raise GLM53ResumeError(f"incorrect physical layer {key}")
                if key in weight_map:
                    raise GLM53ResumeError(f"duplicate checkpoint tensor {key}")
    except (OSError, ValueError) as exc:
        raise GLM53ResumeError(f"corrupt existing safetensors: {filename}") from exc
    for key in keys:
        weight_map[key] = filename
    return True


def expected_expert_keys(physical: int, slot: int) -> set[str]:
    root = f"layers.{physical}.moe.routed_experts.{slot}"
    return {f"{root}.{name}_proj.weight" for name in ("gate", "up", "down")}


def validate_legacy_partial(output: Path) -> None:
    """Explicitly adopt a stopped legacy run. Never use concurrently with writer."""
    if not output.is_dir():
        raise GLM53ResumeError("missing previous checkpoint output directory")
    if (output / "COMPILE_COMPLETE.json").exists():
        raise GLM53ResumeError("cannot resume a completed checkpoint")
    known = {"model.safetensors.index.json", "transplant_manifest.json",
             "iq_config.json", "GLM53_RESUME_MANIFEST.json", "INCOMPLETE"}
    found = 0
    for path in output.iterdir():
        name = path.name
        if name in known or (name.startswith("model-moe-") and
                             name.endswith("-projection.json")):
            continue
        if path.is_symlink() or not path.is_file() or not _SHARD_RE.fullmatch(name):
            raise GLM53ResumeError(f"unexpected partial checkpoint file: {name}")
        found += 1
    if found == 0:
        raise GLM53ResumeError("no partial shards to resume")
