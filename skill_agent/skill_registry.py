from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    root_dir: Path
    scripts_dir: Path
    skill_md: Path


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*"(.*)"\s*$', re.MULTILINE)


def _parse_skill_md_frontmatter(md_text: str) -> tuple[str | None, str | None]:
    m = _FRONTMATTER_RE.match(md_text)
    if not m:
        return None, None
    block = m.group(1)
    kv = dict(_KV_RE.findall(block))
    return kv.get("name"), kv.get("description")


def _iter_skill_roots(project_root: Path) -> Iterable[Path]:
    candidates = [
        project_root / ".trae" / "skills",
        project_root / "skills",
    ]
    for base in candidates:
        if base.is_dir():
            yield base


def load_skills(project_root: str | Path) -> list[Skill]:
    project_root = Path(project_root)
    skills: list[Skill] = []
    for base in _iter_skill_roots(project_root):
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            md = d / "SKILL.md"
            if not md.is_file():
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except Exception:
                continue
            name, desc = _parse_skill_md_frontmatter(text)
            if not name:
                name = d.name
            if not desc:
                desc = ""
            scripts_dir = d / "scripts"
            skills.append(
                Skill(
                    name=str(name),
                    description=str(desc),
                    root_dir=d,
                    scripts_dir=scripts_dir,
                    skill_md=md,
                )
            )
    skills.sort(key=lambda s: s.name)
    return skills


def find_skill(project_root: str | Path, name: str) -> Skill | None:
    for s in load_skills(project_root):
        if s.name == name:
            return s
    return None

