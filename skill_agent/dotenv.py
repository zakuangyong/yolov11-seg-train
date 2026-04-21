from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        return v[1:-1]
    return v


def parse_dotenv(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            continue
        val = _strip_quotes(v.strip())
        out[key] = val
    return out


def apply_dotenv(path: str | Path, *, override: bool = False) -> dict[str, str]:
    env = parse_dotenv(path)
    for k, v in env.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return env

