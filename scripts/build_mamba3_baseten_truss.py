#!/usr/bin/env python3
"""Stage a Baseten Truss for EXACT Mamba-3 + Phase 5e weights, outside Git.

No network requests or deployment. Verifies adapter SHA-256 before copying.
The original Mamba-3 foundation is downloaded/validated by the remote model.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iq_local.mamba3_csa_hca import PHASE5E_SHA256, _sha256
CHECKPOINT = ROOT / "checkpoints/mamba3-csa-hca/iq_phase5e_attention_candidate.pt"

FILES = {
    "deploy/baseten_mamba3_phase5e/config.yaml": "config.yaml",
    "deploy/baseten_mamba3_phase5e/model.py": "model/model.py",
    "iq_local/__init__.py": "packages/iq_local/__init__.py",
    "iq_local/mamba3_csa_hca.py": "packages/iq_local/mamba3_csa_hca.py",
    "experiments/mamba3-csa-hca/iq_mamba3_csa_hca_phase5e/iq_hybrid_cpu.py":
        "packages/experiments/mamba3-csa-hca/iq_mamba3_csa_hca_phase5e/iq_hybrid_cpu.py",
}


def stage(checkpoint: Path, output: Path):
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Truss weights must be staged OUTSIDE the Git repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Truss directory: {output}")
    actual = _sha256(checkpoint)
    if actual != PHASE5E_SHA256:
        raise ValueError(f"Wrong checkpoint SHA-256: {actual}")
    for src, dest in FILES.items():
        path = output / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, path)
    data_path = output / "data" / "iq_phase5e_attention_candidate.pt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, data_path)
    if _sha256(data_path) != PHASE5E_SHA256:
        raise RuntimeError("Staged adapter hash mismatch")
    print(f"Truss staging verified: {output}")
    print(f"Adapter SHA-256: {actual}")
    print("No deployment performed; review Baseten credits/hardware before pushing.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attention-checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--output-dir", type=Path,
                   default=Path.home() / ".cache/iq/phase5e-baseten-truss")
    args = p.parse_args()
    stage(args.attention_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
