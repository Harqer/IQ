"""Validate the complete dependency graph needed by IQ inference.

Do not treat unrelated build tools inherited from the HF container as
inference dependencies. In particular, the system-image package `spin`
requires Click <8.4 whereas Transformers/HF Hub can require a newer Click.
A blanket `pip check` flags that unrelated system package even when IQ's
inference graph is fully consistent.
"""

from __future__ import annotations

import sys
from importlib import metadata

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

# Packages imported or required by the complete IQ inference runtime.
# Walk their transitive non-optional dependencies, not the image's unrelated
# globally installed build/maintenance utilities.
ROOTS = (
    "torch",
    "triton",
    "transformers",
    "safetensors",
    "huggingface-hub",
    "numpy",
    "mamba-ssm",
    "tilelang",
    "apache-tvm-ffi",
    "quack-kernels",
)


def main() -> int:
    pending = list(ROOTS)
    seen: set[str] = set()
    problems: list[str] = []

    while pending:
        name = pending.pop()
        normalized = canonicalize_name(name)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            installed = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: missing from IQ runtime")
            continue
        for raw in installed.requires or ():
            try:
                required = Requirement(raw)
            except InvalidRequirement as exc:
                problems.append(f"{name}: invalid dependency metadata {raw!r}: {exc}")
                continue
            # Extras are not enabled by default. Markers like
            # 'extra == "dev"' must not force training-only packages.
            if required.marker and not required.marker.evaluate({"extra": ""}):
                continue
            pending.append(required.name)
            try:
                dep_version = metadata.version(required.name)
            except metadata.PackageNotFoundError:
                problems.append(f"{name}: missing required {required}")
                continue
            try:
                if required.specifier and Version(dep_version) not in required.specifier:
                    problems.append(
                        f"{name}: requires {required}, installed {required.name}=={dep_version}"
                    )
            except InvalidVersion as exc:
                problems.append(f"{name}: invalid installed version {dep_version!r}: {exc}")

    if problems:
        for item in problems:
            print(f"IQ DEPENDENCY ERROR: {item}", file=sys.stderr)
        return 2
    print(f"IQ inference dependency graph verified ({len(seen)} distributions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
