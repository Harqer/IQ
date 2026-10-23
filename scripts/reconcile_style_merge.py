"""Safely finish IQ's local PR57 + formatting-commit merge.

Reads git's three conflict stages for the two known overlapping files.
Also resolves additive import/export differences in iq_model/__init__.py;
never resolves arbitrary AST changes or changed comments.

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



def _comment_tokens(source: str) -> tuple[str, ...]:
    return tuple(
        tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
    )


def _inspect_init(blob: bytes) -> tuple[str, ast.Module, dict, list[str], str]:
    source = blob.decode("utf-8")
    tree = ast.parse(source)
    imports = {}
    exports = None
    other_nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            key = (node.level, node.module or "")
            if key in imports or any(item.name == "*" for item in node.names):
                raise ValueError(f"duplicate or wildcard import for {key}")
            imports[key] = list(node.names)
        elif isinstance(node, ast.Import):
            # Preserve simple imports exactly; review additions manually.
            other_nodes.append(node)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            if (
                exports is not None
                or not isinstance(node.value, (ast.List, ast.Tuple))
                or any(
                    not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                    for item in node.value.elts
                )
            ):
                raise ValueError("dynamic or duplicate __all__ declaration")
            exports = [item.value for item in node.value.elts]
        else:
            other_nodes.append(node)
    if exports is None:
        raise ValueError("no static __all__ declaration")
    other = ast.dump(ast.Module(body=other_nodes, type_ignores=[]), include_attributes=False)
    return source, tree, imports, exports, other


def _import_aliases(items: list[ast.alias]) -> list[tuple[str, str | None]]:
    return [(item.name, item.asname) for item in items]


def _merge_init_imports_and_exports(base: bytes, ours: bytes, theirs: bytes) -> bytes:
    """Combine only additive import/export edits and formatting safely."""
    b_text, _, b_imports, b_exports, b_other = _inspect_init(base)
    o_text, o_tree, o_imports, o_exports, o_other = _inspect_init(ours)
    t_text, _, t_imports, t_exports, t_other = _inspect_init(theirs)
    if b_other != o_other or b_other != t_other:
        raise ValueError("a side changes Python code beyond imports and __all__")
    if _comment_tokens(b_text) != _comment_tokens(t_text):
        raise ValueError("the formatting side changes comments")
    if _comment_tokens(b_text) != _comment_tokens(o_text):
        raise ValueError("the PR side changes comments")
    if b_imports.keys() - o_imports.keys() or b_imports.keys() - t_imports.keys():
        raise ValueError("a side removed a base import module")
    if t_imports.keys() - o_imports.keys():
        raise ValueError("formatting side adds a new import module; ordering needs review")
    if not set(b_exports).issubset(o_exports) or not set(b_exports).issubset(t_exports):
        raise ValueError("a side removed an existing public export")
    if len(o_exports) != len(set(o_exports)) or len(t_exports) != len(set(t_exports)):
        raise ValueError("duplicate public exports require review")

    edits = []
    lines = o_text.splitlines(keepends=True)
    for node in o_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        key = (node.level, node.module or "")
        base_names = set(_import_aliases(b_imports.get(key, [])))
        ours_names = _import_aliases(o_imports[key])
        theirs_names = _import_aliases(t_imports[key])
        if not base_names.issubset(ours_names) or not base_names.issubset(theirs_names):
            raise ValueError(f"a side removed existing imported names from {key}")
        extras = [item for item in theirs_names if item not in ours_names]
        if not extras:
            continue
        if any("#" in line for line in lines[node.lineno - 1:node.end_lineno]):
            raise ValueError(f"cannot rewrite comments within import {key}")
        merged_names = ours_names + extras
        module = "." * node.level + (node.module or "")
        replacement = f"from {module} import (\n" + "".join(
            f"    {name}{' as ' + alias if alias else ''},\n"
            for name, alias in merged_names
        ) + ")\n"
        edits.append((node.lineno - 1, node.end_lineno, replacement))

    added_exports = [name for name in t_exports if name not in o_exports]
    if added_exports:
        available = {
            alias.asname or alias.name
            for names in list(o_imports.values()) + list(t_imports.values())
            for alias in names
        }
        if not set(added_exports).issubset(available):
            raise ValueError(f"unknown exported names: {set(added_exports) - available}")
        export_node = next(
            node for node in o_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        )
        if any("#" in line for line in lines[export_node.lineno - 1:export_node.end_lineno]):
            raise ValueError("cannot rewrite comments in __all__")
        replacement = "__all__ = [\n" + "".join(
            f"    {name!r},\n" for name in o_exports + added_exports
        ) + "]\n"
        edits.append((export_node.lineno - 1, export_node.end_lineno, replacement))

    for start, end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        lines[start:end] = [replacement]
    merged = "".join(lines).encode("utf-8")
    _, _, merged_imports, merged_exports, merged_other = _inspect_init(merged)
    if merged_other != o_other:
        raise ValueError("merged file changes unexpected Python code")
    if not set(o_exports).issubset(merged_exports) or not set(t_exports).issubset(merged_exports):
        raise ValueError("merged exports incomplete")
    for side in (o_imports, t_imports):
        for key, values in side.items():
            if not set(_import_aliases(values)).issubset(_import_aliases(merged_imports[key])):
                raise ValueError(f"merged imports incomplete: {key}")
    print("INIT: preserved all imported names and public exports from both versions")
    if added_exports:
        print("INIT: added exports:", ", ".join(added_exports))
    return merged


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
        if same_semantics:
            merged = ours
            reason = "formatting side matches base AST and comments"
        elif path == "iq_model/__init__.py":
            try:
                merged = _merge_init_imports_and_exports(base, ours, theirs)
                reason = "verified additive import/export union"
            except ValueError as exc:
                print(f"MANUAL: {path} ({exc})")
                continue
        else:
            print(f"MANUAL: {path} (formatting side changes AST and/or comments)")
            continue

        # Save the unresolved working copy before overwriting conflict markers.
        target = root / path
        backup = report_path.with_suffix(".unresolved-init.txt")
        if path == "iq_model/__init__.py":
            backup.write_bytes(target.read_bytes())
            print(f"Saved original unresolved init file to {backup}")
        target.write_bytes(merged)
        if shutil.which("ruff") is not None:
            fmt = subprocess.run(["ruff", "format", str(target)], capture_output=True, text=True)
            if fmt.returncode:
                target.write_bytes(ours)
                print(f"MANUAL: {path} (ruff failed: {fmt.stderr.strip()})")
                continue
        git("add", "--", path)
        resolved.append(path)
        print(f"RESOLVED: {path} ({reason})")

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
