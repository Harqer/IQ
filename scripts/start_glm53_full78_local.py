#!/usr/bin/env python3
"""Start the genuine 78-layer GLM-5.3 -> IQ/Mamba-3 compilation on local CUDA.

The prior 866-file calibration receipt is independently verified as files are
downloaded, so the 35 freshly computed WARM maps are NEVER recomputed.
Creates an entirely new checkpoint output; --resume uses original provenance.
If --hf-output-mount points to a separately verified official hf-mount FUSE
bucket mount, compiler shards are written remotely without accumulating a
full BF16 checkpoint on the local SSD. Remote completeness still needs an
independent read-only audit before declaring durable success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tempfile
import uuid

from huggingface_hub import HfApi, hf_hub_download
from scripts.assemble_glm53_full78_calibration_local import OUTPUT, PREFIX, RECEIPT, inventory, read
from scripts.glm53_full78_gate import expected_calibration_files, check_checkpoint_complete
from scripts.full78_storage_preflight import required_private_bytes

GLM_REPO = "zai-org/GLM-5.3-BF16"
GLM_REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAMBA_REPO = "state-spaces/mamba3-mimo-1.5b"
MAMBA_REV = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_space(directory: Path, calibration_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(directory).free
    # Count existing files toward the required durable BF16 volume on resume.
    calib = directory / "calibration"
    checkpoints = directory / "IQ" / "runs"
    existing_calib_bytes = sum(p.stat().st_size for p in calib.iterdir() if p.is_file()) if calib.is_dir() else 0
    existing_shard_bytes = sum(
        p.stat().st_size for d in checkpoints.iterdir() if d.is_dir()
        for p in d.iterdir() if p.is_file()
    ) if checkpoints.is_dir() else 0
    required = max(0, required_private_bytes() - existing_shard_bytes) + max(
        0, calibration_bytes - existing_calib_bytes
    ) + 5_000_000_000
    print(
        f"IQ_LOCAL_DISK_FREE_GB={available/1e9:.1f} "
        f"MINIMUM_REQUIRED_GB={required/1e9:.1f}", flush=True
    )
    if available < required:
        raise RuntimeError(
            "Not enough available local disk space to complete BF16 transfer. "
            "Do not start the compiler; use a larger filesystem or implement "
            "verified incremental remote-shard upload/cleanup first."
        )


def checked_calibration(directory: Path, *, remote_output: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    receipt = read(OUTPUT, RECEIPT, directory / "calibration_sha256.json")
    manifest = read(OUTPUT, f"{PREFIX}/glm53_calibration.json",
                    directory / "glm53_calibration.json")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    record = json.loads(receipt.read_text(encoding="utf-8"))
    expected = expected_calibration_files(metadata)
    hashes = record.get("files")
    if (
        record.get("schema_version") != 1
        or record.get("source_layers") != 78
        or record.get("hash") != "sha256"
        or len(expected) != 866
        or not isinstance(hashes, dict)
        or set(hashes) != expected
        or [x.get("source_layer") for x in metadata["stages"]] != list(range(78))
    ):
        raise RuntimeError("Invalid full78 calibration receipt or source-layer inventory")
    actual = inventory(OUTPUT, PREFIX)
    needed = sum(actual.get(f"{PREFIX}/{name}", 0) for name in expected)
    if any(actual.get(f"{PREFIX}/{name}", 0) <= 0 for name in expected):
        raise RuntimeError("Calibration has missing/empty Hugging Face artifacts")
    # The output checkpoint must live on the same volume as the calibration
    # working root for this conservative complete-BF16 disk-space preflight.
    base = directory.parent
    if remote_output:
        # Preserve local space for calibration, pinned Mamba donor and CPU/GPU
        # scratch. The complete BF16 checkpoint is stored on a verified mount.
        local_calib_bytes = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
        needed_local = max(0, needed - local_calib_bytes) + 20_000_000_000
        available_local = shutil.disk_usage(base).free
        print(
            f"IQ_LOCAL_REMOTE_OUTPUT_FREE_GB={available_local / 1e9:.1f} "
            f"MIN_LOCAL_REQUIRED_GB={needed_local / 1e9:.1f}",
            flush=True,
        )
        if available_local < needed_local:
            raise RuntimeError("Insufficient space for local calibration and scratch")
    else:
        require_space(base, needed)

    for i, name in enumerate(sorted(expected), 1):
        remote = f"{PREFIX}/{name}"
        target = directory / name
        if target.exists():
            if not target.is_file() or target.is_symlink():
                raise RuntimeError(f"Untrusted local calibration item: {name}")
        else:
            read(OUTPUT, remote, target)
        if digest(target) != hashes[name]:
            raise RuntimeError(
                f"LOCAL_CALIBRATION_SHA_MISMATCH={name} - stop, do not overwrite"
            )
        if i % 50 == 0 or i == len(expected):
            print(f"IQ_LOCAL_VERIFIED_CALIBRATION={i}/866", flush=True)
    print("IQ_LOCAL_FULL78_CALIBRATION_VERIFIED", flush=True)


def fetch_donors(root: Path) -> tuple[Path, Path]:
    glm = root / "donors" / "glm53"
    mamba = root / "donors" / "mamba3"
    for repo, revision, dest, names in (
        (GLM_REPO, GLM_REV, glm, ("config.json", "model.safetensors.index.json")),
        (MAMBA_REPO, MAMBA_REV, mamba, ("config.json", "pytorch_model.bin")),
    ):
        dest.mkdir(parents=True, exist_ok=True)
        for filename in names:
            path = Path(hf_hub_download(
                repo_id=repo, revision=revision, filename=filename,
                local_dir=str(dest)
            ))
            if path.stat().st_size < 1:
                raise RuntimeError(f"Empty donor file: {repo}/{filename}")
        print(f"IQ_PINNED_DONOR_READY={repo}", flush=True)
    return glm, mamba


def verify_hf_output_mount(mount: Path, run_id: str) -> None:
    """Fail closed unless a real mount maps to the exact intended HF run.

    The probe uses the same create-temp, fsync, rename semantics as the model
    shard writer. A SEPARATE HF API read must see the exact bytes before
    compilers may use the mount. The probe is removed before compilation.
    """
    if not mount.is_dir() or not os.path.ismount(mount):
        raise RuntimeError(
            "HF remote output must be an ACTIVE mountpoint, not a local directory"
        )
    from huggingface_hub import bucket_info
    if bucket_info(OUTPUT).private:
        raise RuntimeError("HF output bucket changed visibility; recheck capacity/policy")
    prefix = f"IQ/runs/{run_id}"
    # Mount the BUCKET ROOT, not an implicit subfolder. The HF Bucket's
    # IQ/runs may be an implicit prefix (no stat'able directory object).
    runs_dir = mount / "IQ" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(exist_ok=True)
    suffix = uuid.uuid4().hex
    temp = run_dir / f".iq-{suffix}.partial"
    final = run_dir / f".iq-{suffix}.probe"
    data = os.urandom(65536)
    try:
        with temp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, final)
        # Advanced-write FUSE flushes asynchronously. Confirm actual remote
        # persistence through an independent Hugging Face client.
        with tempfile.TemporaryDirectory(prefix="iq-hf-mount-verify-") as root:
            destination = Path(root) / "hf-confirmed.bin"
            observed = False
            for attempt in range(30):
                try:
                    read(OUTPUT, f"{prefix}/{final.name}", destination)
                    if destination.read_bytes() != data:
                        raise RuntimeError("HF mount upload SHA mismatch")
                    observed = True
                    break
                except Exception as exc:
                    if isinstance(exc, RuntimeError) and "SHA mismatch" in str(exc):
                        raise
                    # A newly written advanced-FUSE file may still be queued
                    # for remote upload. Retry missing/rate-limited reads,
                    # not authentication or permission failures.
                    error = str(exc).lower()
                    if not any(x in error for x in (
                        "404", "not found", "does not exist", "429", "rate limit",
                    )):
                        raise
                    destination.unlink(missing_ok=True)
                    time.sleep(3)
            if not observed:
                raise RuntimeError(
                    "HF mount created local file, but independent bucket "
                    "readback could not verify its durability; refusing compile"
                )
        print("IQ_HF_OUTPUT_MOUNT_REMOTE_READBACK_VERIFIED", flush=True)
    finally:
        temp.unlink(missing_ok=True)
        final.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    if os.getenv("IQ_CONFIRM_NO_OTHER_WRITER") != "1":
        raise RuntimeError("Set IQ_CONFIRM_NO_OTHER_WRITER=1 after checking no other writer")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{7,79}", args.run_id):
        raise RuntimeError("RUN_ID must be an explicit unique lowercase ID")
    if args.run_id == "glm53-full78-20261024-c":
        raise RuntimeError("Never overwrite or adopt the unprovenanced historical run")
    hf_owner = HfApi().whoami()["name"]
    if hf_owner.casefold() != "kofecheckmate":
        raise RuntimeError(f"Signed into HF as {hf_owner}, expected kofecheckmate")

    import torch
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA torch is required, no silent CPU fallback")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 6.5 * 1024**3:
        raise RuntimeError("This local launcher requires at least 6.5 GiB GPU RAM")
    print(f"IQ_LOCAL_COMPILE_DEVICE={gpu.name} VRAM_GIB={gpu.total_memory/1024**3:.1f}", flush=True)

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    calibration = root / "calibration"
    remote_output = args.hf_output_mount is not None
    mount = args.hf_output_mount.expanduser().resolve() if remote_output else None
    output = (
        mount / "IQ" / "runs" / args.run_id
        if mount is not None else root / "IQ" / "runs" / args.run_id
    )
    if mount is not None:
        verify_hf_output_mount(mount, args.run_id)
    if args.resume:
        if not (output / "run-provenance.json").is_file():
            raise RuntimeError("No provenanced partial local checkpoint; --resume refused")
        if (output / "COMPILE_COMPLETE.json").exists():
            raise RuntimeError("Output already complete; refusing resume")
    elif output.exists() and (
        not remote_output or any(output.iterdir())
    ):
        raise RuntimeError(
            f"Output already contains files: {output}. Use --resume ONLY "
            "for a provenanced run, or choose a fresh run ID."
        )
    if calibration.exists() and calibration.is_symlink():
        raise RuntimeError("Unsafe calibration symlink")
    checked_calibration(calibration, remote_output=remote_output)
    glm, mamba = fetch_donors(root)

    # Limit GPU RAM allocations while processing thousands of genuine experts.
    if os.getenv("IQ_PROJECTION_DEVICE", "cuda") != "cuda":
        raise RuntimeError("IQ_PROJECTION_DEVICE must be cuda")
    os.environ["IQ_PROJECTION_DEVICE"] = "cuda"
    os.environ.setdefault("IQ_MOE_GPU_BATCH_EXPERTS", "2")
    os.environ.setdefault("IQ_MOE_GPU_BATCH_SOURCE_MIB", "256")
    os.environ.setdefault("IQ_MOE_GPU_PREFETCH", "0")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if args.prepare_only:
        print("IQ_LOCAL_COMPILE_PREPARED=1", flush=True)
        print(f"IQ_NEXT_STEP=rerun without --prepare-only to compile into {output}", flush=True)
        return
    command = [
        sys.executable, "-m", "iq_transfer.cli", "glm53-compile",
        "--checkpoint", str(glm),
        "--calibration", str(calibration),
        "--mamba3-checkpoint", str(mamba),
        "--output", str(output),
        "--routed-experts", "256", "--full-78",
        "--checkpoint-revision", GLM_REV,
        "--repo-id", GLM_REPO,
        "--mamba3-revision", MAMBA_REV,
        "--donor-license", "GLM-5.3", "--streaming-source",
    ]
    if args.resume:
        command.append("--resume")
    print(f"IQ_ACTUAL_GLM53_WEIGHT_COMPILATION_START={args.run_id}", flush=True)
    subprocess.run(command, check=True)

    manifest = json.loads((output / "transplant_manifest.json").read_text())
    source_layers = {int(v) for v in manifest["source_layer_map"].values()}
    if source_layers != set(range(78)):
        raise RuntimeError("Incomplete GLM source layer coverage")
    experts = manifest["selected_source_expert_ids"]
    if len(experts) != 78:
        raise RuntimeError("Incorrect source expert stage inventory")
    for layer in range(78):
        slots = experts[str(layer)]
        if layer < 3:
            if slots is not None:
                raise RuntimeError(f"Expected dense-only layer {layer}")
        elif not isinstance(slots, list) or set(slots) != set(range(256)):
            raise RuntimeError(f"Missing real donor experts in layer {layer}")
    if (output / "COMPILE_COMPLETE.json").exists():
        raise RuntimeError("Completion marker already exists; refusing overwrite")

    complete = {
        "status": "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED",
        "source_layers": 78,
        "mapped_layers": 78,
        "routed_source_layers": 75,
        "routed_experts_per_layer": 256,
        "mapped_routed_expert_slots": 19200,
        "not_exact_bitwise_weight_reuse": True,
        "tokenizer_packaged": all(
            (output / n).is_file() for n in
            ("tokenizer.json", "tokenizer_config.json")
        ),
        "inference_ready": False,
        "end_to_end_knowledge_retention_proven": False,
    }
    temp = output / "COMPILE_COMPLETE.json.partial"
    with temp.open("x", encoding="utf-8") as fp:
        json.dump(complete, fp, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    from scripts.glm53_full78_gate import expected_stage_shards
    physical = set()
    for stage_number in range(78):
        physical.update(expected_stage_shards(stage_number))
    physical.update(("model-global.safetensors", "model-native.safetensors",
                     "transplant_manifest.json", "iq_config.json"))
    physical.update(f"model-mamba-{n:03d}.safetensors" for n in range(79))
    missing = [
        name for name in sorted(physical) if not (output / name).is_file()
        or (output / name).stat().st_size < 1
    ]
    if missing:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Output missing {len(missing)} checkpoint files: {missing[:8]}")
    os.replace(temp, output / "COMPILE_COMPLETE.json")
    status = check_checkpoint_complete(output)
    if not status["ready"]:
        raise RuntimeError("Full compiler exited, but expected checkpoint shards are missing")
    if remote_output:
        print(
            f"IQ_FULL78_LOCAL_COMPILE_ENDED_PENDING_REMOTE_DURABILITY_AUDIT="
            f"{args.run_id}", flush=True,
        )
    else:
        print(f"IQ_FULL78_LOCAL_COMPILE_COMPLETE={args.run_id} inference_ready=False", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.home() / "IQ-transfer-full78")
    p.add_argument("--run-id", default="glm53-full78-local-20261025a")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--hf-output-mount", type=Path,
        help="ACTIVE official hf-mount FUSE mount of the "
             "kofecheckmate/iq-glm53-all256 BUCKET ROOT; "
             "requires external HF storage capacity/fair-use agreement",
    )
    p.add_argument("--prepare-only", action="store_true")
    options = p.parse_args()
    if options.resume and options.prepare_only:
        p.error("--resume and --prepare-only are mutually exclusive")
    run(options)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"IQ_LOCAL_COMPILE_BLOCKED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3) from exc
