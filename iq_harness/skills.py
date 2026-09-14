from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    root: Path
    body: str
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    @property
    def skill_file(self) -> Path:
        return self.root / "SKILL.md"

    def read_resource(self, relative_path: str) -> str:
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise SkillError(f"resource escapes skill root: {relative_path}")
        if not candidate.is_file():
            raise SkillError(f"skill resource not found: {relative_path}")
        return candidate.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillError("SKILL.md frontmatter is not terminated") from exc
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1 :]).strip()


def load_skill(skill_dir: str | Path) -> Skill:
    root = Path(skill_dir).resolve()
    path = root / "SKILL.md"
    if not path.is_file():
        raise SkillError(f"missing SKILL.md in {root}")

    frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(frontmatter) or {}
    if not isinstance(data, dict):
        raise SkillError("SKILL.md frontmatter must be a mapping")

    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or len(name) > 64:
        raise SkillError("skill name must be 1-64 chars of lowercase letters, numbers, and hyphens")
    if root.name != name:
        raise SkillError(f"skill folder must match skill name: expected {name!r}, got {root.name!r}")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise SkillError("skill description must be a non-empty string up to 1024 chars")

    compatibility = data.get("compatibility")
    if compatibility is not None and (not isinstance(compatibility, str) or len(compatibility) > 500):
        raise SkillError("compatibility must be a string up to 500 chars")

    allowed = data.get("allowed-tools", "")
    if allowed is None:
        allowed = ""
    if not isinstance(allowed, str):
        raise SkillError("allowed-tools must be a space-separated string")

    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise SkillError("metadata must be a mapping")

    return Skill(
        name=name,
        description=description.strip(),
        root=root,
        body=body,
        license=data.get("license") if isinstance(data.get("license"), str) else None,
        compatibility=compatibility,
        allowed_tools=tuple(part for part in allowed.split() if part),
        metadata=metadata,
    )


class SkillCatalog:
    """Discovers Agent Skills and exposes progressive-disclosure metadata."""

    def __init__(self, roots: Iterable[str | Path] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for root in roots:
            self.discover(root)

    def discover(self, root: str | Path) -> None:
        base = Path(root).resolve()
        if not base.exists():
            return
        candidates = [base] if (base / "SKILL.md").is_file() else [p for p in base.iterdir() if p.is_dir()]
        for candidate in sorted(candidates):
            if not (candidate / "SKILL.md").is_file():
                continue
            skill = load_skill(candidate)
            if skill.name in self._skills:
                raise SkillError(f"duplicate skill name: {skill.name}")
            self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillError(f"unknown skill: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def catalog_text(self, allowed: Iterable[str] | None = None) -> str:
        names = set(allowed) if allowed else set(self._skills)
        rows = ["<available_skills>"]
        for name in sorted(names):
            if name not in self._skills:
                continue
            skill = self._skills[name]
            rows.extend([
                "  <skill>",
                f"    <name>{skill.name}</name>",
                f"    <description>{skill.description}</description>",
                "  </skill>",
            ])
        rows.append("</available_skills>")
        return "\n".join(rows)
