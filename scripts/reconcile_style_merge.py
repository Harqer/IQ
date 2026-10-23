"""Safely finish IQ's local PR57 + formatting-commit merge.

Reads git's three conflict stages for the two known overlapping files.
Only resolves automatically if the formatting side differs from the merge
base solely in formatting: same Python AST and same comment tokens.

Run INSIDE the pending ~/IQ-integrate worktree merge. Does not push or
change the remote main branch. Saves a full conflict review outside repo.
"""

from __future__ import annotations

import ast
import difflib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tokenize


KNOWN = {
    "iq_model/__init__.py",
    "iq_model/attention/compressed.py",
}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], check=check, capture_output=True)


def stage(index: int, path: str) -> bytes:
    return git("show", f":{index}:{path}").stdout


def semantic_key(blob: bytes) -> tuple[str, tuple[str, ...]]:
    source = blob.decode("utf-8")
    syntax = ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)
    comments = tuple(
        token.string.strip()
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    )
    return syntax, comments


def run() -> int:
    if not git("rev-parse", "--verify", "MERGE_HEAD", check=False).returncode == 0:
        print("This helper requires an active git merge (MERGE_HEAD absent).")
        return 2

    root = Path(git("rev-parse", "--show-toplevel").stdout.decode().strip())
    paths = git("diff", "--name-only", "--diff-filter=U", "-z").stdout.decode().split("\0")
    conflicts = [p for p in paths if p]
    if not conflicts:
        print("No unresolved Git conflicts; nothing to resolve.")
        return 0

    # Create a durable review record BEFORE making any change.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="IQ-merge-review-", suffix=".txt",
        dir=Path.home(), delete=False,
    ) as report:
        report_path = Path(report.name)
        report.write("Git unresolved conflicts (before automatic reconciliation)\n\n")
        report.write("Worktree: " + str(root) + "\n")
        report.write("Conflicts: " + ", ".join(conflicts) + "\n\n")
        report.write(git("diff", "--cc", "--", *conflicts, check=False).stdout.decode(
            "utf-8", errors="replace"
        ))
        for path in conflicts:
            try:
                base, ours, theirs = (stage(i, path) for i in (1, 2, 3))
            except subprocess.CalledProcessError:
                report.write(f"\n{path}: missing merge stage, requires manual resolution\n")
                continue
            report.write(f"\n==== {path}: common ancestor vs formatting commit ====\n")
            report.writelines(
                difflib.unified_diff(
                    base.decode("utf-8", errors="replace").splitlines(True),
                    theirs.decode("utf-8", errors="replace").splitlines(True),
                    fromfile=f"common-ancestor:{path}", tofile=f"formatting-side:{path}",
                )
            )
            report.write(f"\n==== {path}: ancestor vs PR57 implementation ====\n")
            report.writelines(
                difflib.unified_diff(
                    base.decode("utf-8", errors="replace").splitlines(True),
                    ours.decode("utf-8", errors="replace").splitlines(True),
                    fromfile=f"common-ancestor:{path}", tofile=f"PR57-side:{path}",
                )
            )

    print(f"Saved actual two-sided conflict diffs to {report_path}")
    if any(p not in KNOWN for p in conflicts):
        print("Unexpected conflict paths: refusing all automatic resolution.")
        return 3

    resolved = []
    for path in conflicts:
        try:
            base, ours, theirs = (stage(i, path) for i in (1, 2, 3))
            same_semantics = semantic_key(base) == semantic_key(theirs)
        except (subprocess.CalledProcessError, UnicodeDecodeError, SyntaxError, tokenize.TokenError) as exc:
            print(f"MANUAL: {path} (cannot establish formatting-only change: {exc})")
            continue
        if not same_semantics:
            print(f"MANUAL: {path} (formatting side changes AST and/or comments)")
            continue

        # Preserve the PR57 code exactly; only reformat after restoring it.
        target = root / path
        target.write_bytes(ours)
        if shutil.which("ruff") is not None:
            fmt = subprocess.run(["ruff", "format", str(target)], capture_output=True, text=True)
            if fmt.returncode:
                target.write_bytes(ours)
                print(f"MANUAL: {path} (ruff failed: {fmt.stderr.strip()})")
                continue
        git("add", "--", path)
        resolved.append(path)
        print(f"RESOLVED: {path} (formatting side AST and comments equal merge base)")

    remaining = [
        p for p in git("diff", "--name-only", "--diff-filter=U", "-z").stdout.decode().split("\0")
        if p
    ]
    if remaining:
        print("Still unresolved:", ", ".join(remaining))
        print("Review original full diffs:", report_path)
        return 4

    if git("diff", "--cached", "--check", check=False).returncode:
        print("Staged whitespace errors; leaving merge uncommitted.")
        return 5
    syntax = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "iq_model", "iq_transfer", "scripts", "tests"],
        cwd=root,
    )
    if syntax.returncode:
        print("Python syntax check failed; leaving merge uncommitted.")
        return 6

    # Don't auto-commit if there are unstaged edits outside the expected merge.
    dirty = git("diff", "--name-only", "-z").stdout.decode().split("\0")
    if any(p for p in dirty if p):
        print("Unstaged edits detected; leaving merge uncommitted.")
        return 7

    commit = git("commit", "-m", "merge: preserve PR57 optimizations and verified style corrections", check=False)
    if commit.returncode:
        print(commit.stderr.decode(errors="replace"))
        print("Git could not finalize the merge; changes remain staged.")
        return 8

    print("LOCAL MERGE COMMITTED:", git("rev-parse", "--short", "HEAD").stdout.decode().strip())
    print("REVIEW DIFFS:", report_path)
    print("No remote push performed. Fetch/merge origin/feat before updating main.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
