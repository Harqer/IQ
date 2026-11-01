"""Reject any pinned Mamba3 source edits beyond the exact IQ allowlist."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ALLOWED = {
    "pyproject.toml",
    "setup.py",
    "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py",
}


def verify_diff_paths(checkout: Path) -> set[str]:
    raw = subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "--name-only", "--"],
        text=True,
    )
    changed = set(raw.splitlines())
    if changed != ALLOWED or len(raw.splitlines()) != len(ALLOWED):
        raise RuntimeError(
            f"Unexpected Mamba source edits: got={sorted(changed)}, "
            f"allowed={sorted(ALLOWED)}"
        )
    return changed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/verify_iq_mamba_patch_paths.py CHECKOUT")
    changed = verify_diff_paths(Path(sys.argv[1]))
    print("IQ_MAMBA_SOURCE_PATCH_ALLOWLIST_PASS=" + ",".join(sorted(changed)), flush=True)
