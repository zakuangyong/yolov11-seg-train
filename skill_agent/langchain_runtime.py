from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str | None


def load_llm_config() -> LLMConfig:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("LLM_API_KEY")
    api_key = api_key.strip() if api_key else None
    if not base_url:
        base_url = "http://127.0.0.1:8000/v1"
    if not model:
        model = "Qwen3.5-35B"
    return LLMConfig(base_url=base_url, model=model, api_key=api_key)


def build_agent(project_root: str):
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import PromptTemplate
    except Exception as e:
        raise RuntimeError(
            "缺少 langchain 依赖。请安装: langchain langchain-openai langchain-core"
        ) from e

    from .executor import list_skill_scripts, run_skill_script
    from .skill_registry import find_skill, load_skills
    from .dotenv import apply_dotenv

    apply_dotenv(os.path.join(project_root, ".env"), override=False)

    cfg = load_llm_config()
    llm = ChatOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "EMPTY",
        model=cfg.model,
        temperature=0,
    )

    @tool
    def list_skills() -> str:
        skills = load_skills(project_root)
        if not skills:
            return "未发现任何技能(SKILL.md)。"
        lines = []
        for s in skills:
            d = s.description.strip()
            lines.append(f"- {s.name}: {d}" if d else f"- {s.name}")
        return "\n".join(lines)

    @tool
    def list_skill_scripts_tool(skill_name: str) -> str:
        s = find_skill(project_root, skill_name)
        if s is None:
            return f"技能不存在: {skill_name}"
        scripts = list_skill_scripts(s)
        if not scripts:
            return f"技能 {skill_name} 没有 scripts/*.py"
        return "\n".join([p.name for p in scripts])

    @tool
    def run_skill(skill_name: str, script: str, args: str = "") -> str:
        s = find_skill(project_root, skill_name)
        if s is None:
            return f"技能不存在: {skill_name}"
        rr = run_skill_script(skill=s, script=script, args=args, cwd=project_root, timeout_s=1800)
        out = rr.stdout.strip()
        err = rr.stderr.strip()
        parts = [f"returncode={rr.returncode}"]
        if out:
            parts.append("stdout:\n" + out)
        if err:
            parts.append("stderr:\n" + err)
        return "\n\n".join(parts)

    tools = [list_skills, list_skill_scripts_tool, run_skill]

    prompt = PromptTemplate.from_template(
        "你是一个技能调度助手。你可以通过工具列出技能、列出技能脚本、运行技能脚本。\n"
        "当用户提出任务时：先选择最合适的 skill，再选择该 skill 的脚本与参数，调用 run_skill。\n"
        "只在必要时向用户提 1 个澄清问题，否则直接执行。\n\n"
        "可用工具:\n{tools}\n\n"
        "工具使用格式:\n{tool_names}\n\n"
        "用户问题:\n{input}\n\n"
        "{agent_scratchpad}"
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
