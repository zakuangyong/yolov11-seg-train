from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .executor import list_skill_scripts, run_skill_script
from .dotenv import apply_dotenv
from .skill_registry import find_skill, load_skills


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_list(args) -> int:
    skills = load_skills(_project_root())
    for s in skills:
        d = s.description.strip()
        print(f"{s.name}\t{d}" if d else s.name)
    return 0


def cmd_scripts(args) -> int:
    s = find_skill(_project_root(), args.skill)
    if s is None:
        print(f"skill not found: {args.skill}")
        return 2
    for p in list_skill_scripts(s):
        print(p.name)
    return 0


def cmd_run(args) -> int:
    s = find_skill(_project_root(), args.skill)
    if s is None:
        print(f"skill not found: {args.skill}")
        return 2
    rr = run_skill_script(skill=s, script=args.script, args=args.args, cwd=_project_root())
    print(json.dumps({"returncode": rr.returncode}, ensure_ascii=False))
    if rr.stdout:
        print(rr.stdout, end="" if rr.stdout.endswith("\n") else "\n")
    if rr.stderr:
        print(rr.stderr, end="" if rr.stderr.endswith("\n") else "\n")
    return int(rr.returncode)


def cmd_chat(args) -> int:
    if args.llm_base_url:
        os.environ["LLM_BASE_URL"] = args.llm_base_url
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key

    from .langchain_runtime import build_agent

    agent = build_agent(str(_project_root()))
    res = agent.invoke({"input": args.prompt})
    out = res.get("output")
    if out is not None:
        print(out)
    return 0


def main() -> None:
    apply_dotenv(_project_root() / ".env", override=False)

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    p_scripts = sub.add_parser("scripts")
    p_scripts.add_argument("skill")
    p_scripts.set_defaults(func=cmd_scripts)

    p_run = sub.add_parser("run")
    p_run.add_argument("skill")
    p_run.add_argument("script")
    p_run.add_argument("--args", default="")
    p_run.set_defaults(func=cmd_run)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("prompt")
    p_chat.add_argument("--llm_base_url", default=None)
    p_chat.add_argument("--llm_model", default=None)
    p_chat.add_argument("--llm_api_key", default=None)
    p_chat.set_defaults(func=cmd_chat)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
