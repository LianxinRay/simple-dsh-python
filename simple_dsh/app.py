"""Composition entry point: assemble a ready-to-run agent context.

Mirrors what a ``cordis.yml`` profile does upstream — one function that
mounts the services, tools, and guards in dependency order and hands back a
context from which agents are created.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent
from .cordis import Context
from .credentials import Credentials
from .guard import RepeatCallGuard, register_timeout
from .llm import DeepSeekAdapter, LlmRegistry
from .prompts import PromptSection, SystemPrompt
from .session import JsonlSink, Session
from .tools import ToolRegistry
from .tools.fs_tools import register_fs_tools
from .tools.shell import register_shell_tool


def create_app(
    workdir: str | Path,
    *,
    env_path: str | Path | None = None,
    model: str = "deepseek-chat",
    tool_timeout: float = 120.0,
    repeat_call_limit: int = 3,
    log_path: str | Path | None = None,
) -> Context:
    """Assemble the full harness context rooted at ``workdir``.

    Services mounted: ``credentials``, ``llm`` (with the DeepSeek adapter),
    ``systemPrompt``, ``tools`` (fs tools + shell), plus the timeout and
    repeat-call guards. ``DEEPSEEK_API_KEY`` resolves env-over-``.env`` and
    fails loud when absent.
    """
    ctx = Context()
    credentials = Credentials(env_path)
    ctx.service("credentials", credentials)

    llm = LlmRegistry()
    base_url = credentials.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    llm.register_adapter(
        DeepSeekAdapter(credentials.require("DEEPSEEK_API_KEY"), base_url=base_url)
    )
    ctx.service("llm", llm)

    prompt = SystemPrompt()
    prompt.register_section(PromptSection(
        id="role", priority=10,
        render=lambda: "You are a careful coding assistant.",
    ))
    prompt.register_section(PromptSection(
        id="workspace", priority=50,
        render=lambda: f"The workspace root is {Path(workdir).resolve()}. "
                       "All file paths are relative to it.",
    ))
    ctx.service("systemPrompt", prompt)

    tools = ToolRegistry(ctx)
    ctx.service("tools", tools)
    register_fs_tools(tools, workdir)
    register_shell_tool(tools, workdir)
    RepeatCallGuard(limit=repeat_call_limit).register(tools)
    register_timeout(ctx, tool_timeout)

    ctx.service("_app_config", {
        "model": model,
        "log_path": str(log_path) if log_path else None,
        "workdir": str(Path(workdir).resolve()),
    })
    return ctx


def create_agent(ctx: Context) -> Agent:
    """Create an agent on an assembled context, wiring the session sink."""
    config = ctx.service("_app_config")
    session = Session()
    sink = None
    if config["log_path"]:
        sink = JsonlSink(config["log_path"])
        session.add_sink(sink)
    agent = Agent(ctx, session=session, model=config["model"])
    agent._sink = sink  # kept alive with the agent; close on dispose if needed
    return agent


__all__ = ["create_agent", "create_app"]
