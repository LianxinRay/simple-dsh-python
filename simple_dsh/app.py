"""Composition entry point: assemble a ready-to-run agent context.

Mirrors what a ``cordis.yml`` profile does upstream — one function that
mounts the services, tools, and guards in dependency order and hands back a
context from which agents are created. Composition is driven by a preset
(see :mod:`simple_dsh.preset`); explicit keyword arguments override it.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent
from .approval import ApprovalService, Responder, require_approval
from .compaction import CompactionService, LlmSummarizer
from .cordis import Context
from .credentials import Credentials
from .guard import RepeatCallGuard, register_timeout
from .llm import DeepSeekAdapter, LlmRegistry
from .preset import DEFAULT_PRESET, load_preset, merge_preset
from .prompts import PromptSection, SystemPrompt
from .session import JsonlSink, Session, SqliteSink
from .tools import ToolRegistry
from .tools.fs_tools import register_fs_tools
from .tools.shell import register_shell_tool
from .tools.subagent import register_subagent_tool
from .tools.todo import register_todo_tool
from .tools.web import register_web_tool


def create_app(
    workdir: str | Path,
    *,
    preset: dict | str | Path | None = None,
    env_path: str | Path | None = None,
    model: str | None = None,
    tool_timeout: float | None = None,
    repeat_call_limit: int | None = None,
    approval_responder: Responder | None = None,
    log_path: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> Context:
    """Assemble the full harness context rooted at ``workdir``.

    ``preset`` is a dict or a JSON file path merged over
    :data:`DEFAULT_PRESET`; explicit keyword arguments win over the preset.
    ``DEEPSEEK_API_KEY`` resolves env-over-``.env`` and fails loud when
    absent. Approval is fail-closed: gated tools are denied unless
    ``approval_responder`` allows them.
    """
    if preset is None:
        spec = dict(DEFAULT_PRESET)
    elif isinstance(preset, dict):
        spec = merge_preset(preset)
    else:
        spec = load_preset(preset)
    if model is not None:
        spec["model"] = model
    if tool_timeout is not None:
        spec["guards"]["timeout"] = tool_timeout
    if repeat_call_limit is not None:
        spec["guards"]["repeat_call_limit"] = repeat_call_limit

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
    enabled = spec["tools"]
    if enabled["fs"]:
        register_fs_tools(tools, workdir)
    if enabled["shell"]:
        register_shell_tool(tools, workdir)
    if enabled["web"]:
        register_web_tool(tools)
    if enabled["subagent"]:
        register_subagent_tool(tools, ctx, model=spec["model"])

    RepeatCallGuard(limit=spec["guards"]["repeat_call_limit"]).register(tools)
    register_timeout(ctx, spec["guards"]["timeout"])

    approval = ApprovalService(approval_responder)
    ctx.service("approval", approval)
    if spec["approval"]["tools"]:
        require_approval(ctx, approval, spec["approval"]["tools"])

    if spec["compaction"]["enabled"]:
        ctx.service("compaction", CompactionService(
            LlmSummarizer(llm, spec["model"]),
            max_tokens=spec["compaction"]["max_tokens"],
            keep_recent=spec["compaction"]["keep_recent"],
        ))

    ctx.service("_app_config", {
        "model": spec["model"],
        "log_path": str(log_path) if log_path else None,
        "sqlite_path": str(sqlite_path) if sqlite_path else None,
        "workdir": str(Path(workdir).resolve()),
        "preset": spec,
    })
    return ctx


def create_agent(ctx: Context) -> Agent:
    """Create an agent on an assembled context, wiring the session sinks.

    ``todo_write`` is registered per agent (its state is log-backed by that
    agent's session). Compaction, when enabled, runs before every step.
    """
    config = ctx.service("_app_config")
    session = Session()
    sinks = []
    if config["log_path"]:
        sink = JsonlSink(config["log_path"])
        session.add_sink(sink)
        sinks.append(sink)
    if config["sqlite_path"]:
        sink = SqliteSink(config["sqlite_path"])
        session.add_sink(sink)
        sinks.append(sink)
    agent = Agent(ctx, session=session, model=config["model"])
    register_todo_tool(ctx.service("tools"), session)

    compaction = ctx._lookup_service("compaction")
    if compaction is not None:

        async def maybe_compact(claimed, next_):
            await compaction.maybe_compact(session)
            return await next_()

        agent.ctx.on("agent/pre-step", maybe_compact)

    agent._sinks = sinks  # kept alive with the agent
    return agent


__all__ = ["create_agent", "create_app"]
