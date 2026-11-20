"""Build the Baseten model archive source tree without flattening packages.

Baseten's external_package_dirs copies each *directory's contents* directly
under packages/. Combining several package roots caused duplicate
packages/__init__.py. Stage selected code under one common parent instead.

Only reviewed .py and pinned .json source is copied. Never include secrets,
weight shards, local environment files, or an entire working repository.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "deploy" / "_iq235_baseten_packages"
PACKAGE_NAMES = ("iq_cloud", "iq_model", "scripts", "configs")
ROOT_FILES = {
    "iq_cloud": (
        "__init__.py", "full_iq235.py", "glm53_lexical.py",
        "lexical_warmstart.py", "checkpoint_metadata.py",
        "checkpoint_migration.py", "remote_shard_loader.py",
        "full235_weight_loader.py", "inference_diagnostics.py",
    ),
    "scripts": ("__init__.py", "patch_iq_native_mimo_tail.py"),
    "configs": (
        "__init__.py",
        "iq235_checkpoint_source.json",
        "iq235_glm53_tokenizer_phase_a.json",
        "iq235_source_compile_status.json",
    ),
}


class BasetenStageError(RuntimeError):
    pass


def selected_sources(root: Path = REPO) -> dict[Path, Path]:
    """Map safe, explicitly scoped source files to their staged paths."""
    result: dict[Path, Path] = {}
    for package in PACKAGE_NAMES:
        parent = root / package
        if not parent.is_dir() or parent.is_symlink():
            raise BasetenStageError(f"required package directory missing/unsafe: {parent}")
        if package == "iq_model":
            files = sorted(p for p in parent.rglob("*.py") if p.is_file())
        else:
            files = [parent / filename for filename in ROOT_FILES[package]]
        for source in files:
            if (not source.is_file() or source.is_symlink()
                    or any(p.is_symlink() for p in source.parents if p != root)
                    or source.suffix not in {".py", ".json"}):
                raise BasetenStageError(f"missing or unsafe source path: {source}")
            relative = source.relative_to(root)
            if relative.parts[0] != package or relative in result:
                raise BasetenStageError(f"invalid stage path: {relative}")
            result[relative] = source
    if len(result) < 15:
        raise BasetenStageError("incomplete model source inventory")
    return result


def stage_sources(root: Path = REPO) -> dict[str, object]:
    """Create one common parent containing distinct import package roots."""
    source_map = selected_sources(root)
    dest = root / "deploy" / "_iq235_baseten_packages"
    deploy = dest.parent
    deploy.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".iq235-stage-", dir=deploy))
    inventory = {}
    try:
        for relative, source in source_map.items():
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inventory[relative.as_posix()] = sha256(target.read_bytes()).hexdigest()
        # The manifest is harmless data; it is intentionally not an executable.
        (work / "stage_manifest.json").write_text(
            json.dumps({"files": inventory}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or not dest.is_dir():
                raise BasetenStageError("refusing to replace non-directory/linked stage")
            # Replace only our generated artifact and never another directory.
            if not (dest / "stage_manifest.json").is_file():
                raise BasetenStageError("existing stage lacks marker; refusing removal")
            shutil.rmtree(dest)
        work.rename(dest)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return {
        "baseten_package_parent": str(dest),
        "packaged_source_files": len(inventory),
        "archive_package_roots": list(PACKAGE_NAMES),
        "gpu_started": False,
    }


if __name__ == "__main__":
    print(json.dumps(stage_sources(), indent=2))
