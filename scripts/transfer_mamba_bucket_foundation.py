#!/usr/bin/env python3
"""Publish IQ's pinned, widened Mamba-3 MIMO foundation to its HF Bucket.

Requires the repository's model dependencies, the Hugging Face CLI, Kaggle
internet access, and an authenticated token with bucket read/write permission.
This produces a Mamba foundation overlay, NOT a runnable full hybrid model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_BUCKET = "hf://buckets/kofecheckmate/mocha"
DESTINATION = SOURCE_BUCKET + "/iq-mamba-foundation/"
PINNED_REVISION = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
PINNED_SHA256 = "139c09e9728d1f10ac7f4354d2ae29aaca41851a954aedd8f6728c05aeb6c744"
REPO_ROOT = Path(__file__).resolve().parents[1]


def run(*argv: str, cwd: Path | None = None) -> None:
    subprocess.run(list(argv), check=True, cwd=cwd)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_compilation(root: Path) -> list[Path]:
    manifest_path = root / "mamba3_transplant.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "iq_mamba3_foundation":
        raise ValueError("Unexpected IQ Mamba foundation artifact_type")
    donor = manifest.get("donor", {})
    if (donor.get("revision") != PINNED_REVISION
            or donor.get("checkpoint_sha256") != PINNED_SHA256):
        raise ValueError("Compiled foundation donor provenance mismatch")
    target = manifest.get("target", {}).get("mamba3", {})
    if (target.get("d_model") != 4096 or target.get("num_layers") != 32
            or target.get("mimo_rank") != 4):
        raise ValueError("Compiled IQ target is not 4096-wide rank-4 MIMO x32")
    placements = manifest.get("placements", [])
    if len(placements) != 24:
        raise ValueError("Expected exactly 24 pretrained MIMO placements")
    ordinals = [p["target_mamba_ordinal"] for p in placements]
    if (len(set(ordinals)) != 24 or any(not 0 <= i < 32 for i in ordinals)
            or ordinals != sorted(ordinals)):
        raise ValueError("Invalid MIMO layer placement")
    identity = manifest.get("identity_mamba_ordinals", [])
    if (len(identity) != 8 or len(set(identity)) != 8
            or set(ordinals) | set(identity) != set(range(32))
            or set(ordinals) & set(identity)):
        raise ValueError("Identity layer coverage is not 8/32")
    shards = sorted((root / "mamba").glob("*.safetensors"))
    expected = {root / str(p["shard"]) for p in placements}
    if set(shards) != expected:
        raise ValueError("MIMO shard files do not match manifest placements")
    files = [root / "model-global.safetensors", *shards, manifest_path]
    for file in files:
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Missing or empty foundation file: {file}")
    return files


def write_checksums(root: Path, files: list[Path]) -> Path:
    result = root / "SHA256SUMS"
    result.write_text(
        "".join(f"{digest(f)}  {f.relative_to(root).as_posix()}\n" for f in files),
        encoding="utf-8",
    )
    return result


def verify_uploaded(destination: str, root: Path, files: list[Path]) -> None:
    """Round-trip every uploaded file; do not trust a successful sync alone."""
    with tempfile.TemporaryDirectory(prefix="iq-mimo-readback-") as temp:
        check = Path(temp)
        for local in files:
            relative = local.relative_to(root).as_posix()
            downloaded = check / relative
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            run("hf", "buckets", "cp", destination + relative, str(downloaded))
            if digest(local) != digest(downloaded):
                raise ValueError(f"Remote checksum mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working-dir", type=Path,
        default=Path("/kaggle/working"),
        help="Compute scratch directory; all permanent model files go to HF Bucket.",
    )
    args = parser.parse_args()
    if shutil.which("hf") is None:
        parser.error("Hugging Face CLI missing: https://hf.co/cli/install.sh")
    if not (REPO_ROOT / "iq_transfer" / "cli.py").is_file():
        parser.error("Run from a checkout of the IQ repository")
    working = args.working_dir.resolve()
    working.mkdir(parents=True, exist_ok=True)
    donor_dir = working / "iq-mamba-donor"
    output = working / "iq-mamba-foundation"
    donor_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        parser.error(
            f"Refusing to overwrite an existing foundation directory: {output}"
        )
    # Source files were confirmed in the private bucket by the account owner.
    # Explicit copies avoid recursively pulling previously published IQ output.
    run("hf", "auth", "whoami")
    for name in ("config.json", "pytorch_model.bin"):
        run(
            "hf", "buckets", "cp",
            SOURCE_BUCKET + "/" + name,
            str(donor_dir / name),
        )
    if digest(donor_dir / "pytorch_model.bin") != PINNED_SHA256:
        raise ValueError("Bucket donor differs from IQ's pinned Mamba-3 MIMO model")
    run(
        sys.executable, "-m", "iq_transfer.cli", "mamba3-direct",
        "--checkpoint", str(donor_dir),
        "--output", str(output),
        "--checkpoint-revision", PINNED_REVISION,
        cwd=REPO_ROOT,
    )
    files = validate_compilation(output)
    checksums = write_checksums(output, files)
    files.append(checksums)
    # No --delete: preserve the original donor weights at the bucket root.
    run("hf", "buckets", "sync", str(output), DESTINATION)
    verify_uploaded(DESTINATION, output, files)
    print(
        "PASS: IQ's 24 pretrained Mamba-3 MIMO layers and widened global "
        f"weights are saved and SHA-256 verified at {DESTINATION}"
    )
    print("This is the foundation overlay, not yet a full IQ hybrid checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
