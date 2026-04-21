from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .skill_registry import Skill


@dataclass(frozen=True)
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def _python_exe() -> str:
    return sys.executable or "python"


def list_skill_scripts(skill: Skill) -> list[Path]:
    if not skill.scripts_dir.is_dir():
        return []
    return sorted([p for p in skill.scripts_dir.glob("*.py") if p.is_file()])


def run_skill_script(
    *,
    skill: Skill,
    script: str,
    args: str = "",
    cwd: str | Path | None = None,
    timeout_s: float | None = None,
) -> RunResult:
    scripts = {p.name: p for p in list_skill_scripts(skill)}
    if script not in scripts:
        raise FileNotFoundError(f"脚本不存在或不在技能 scripts 目录中: {script}")

    script_path = scripts[script]
    cmd = [_python_exe(), str(script_path)]
    if args.strip():
        cmd += shlex.split(args, posix=False)

    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        encoding="utf-8",
        errors="replace",
    )
    return RunResult(cmd=cmd, returncode=int(p.returncode), stdout=p.stdout, stderr=p.stderr)

