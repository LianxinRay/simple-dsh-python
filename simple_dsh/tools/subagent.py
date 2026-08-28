"""Subagent delegation tool (P2).

Ported from ``packages/subagent``: a model-facing ``delegate`` tool that
spawns a fresh child agent — its own session log on the same context — runs
its task to completion, and returns the child's final assistant text.
"""

from __future__ import annotations

from ..cordis import Context, Disposer
from ..llm import TextBlock
from .registry import ToolDefinition, ToolRegistry, ToolResult


def register_subagent_tool(
    tools: ToolRegistry,
    ctx: Context,
    *,
    name: str = "delegate",
    model: str | None = None,
) -> Disposer:
    """Register the delegation tool; returns its disposer.

    The child shares the context (tools, llm, prompt sections) but gets its
    own session, so its history never pollutes the parent's.
    """

    async def delegate(args):
        # Deferred import: tools must not import the agent layer at load time.
        from ..agent import Agent

        child = Agent(ctx, model=args.get("model") or model or "default")
        await child.prompt(args["task"])
        await child.when_idle()
        final_text = ""
        for message in reversed(child.session.derive_messages()):
            if message.role == "assistant":
                final_text = "".join(
                    b.text for b in message.content if isinstance(b, TextBlock)
                )
                break
        steps = sum(1 for e in child.session if e.type == "step/start")
        return ToolResult(
            content=[TextBlock(text=final_text or "(subagent produced no output)")],
            meta={"subagent_steps": steps, "subagent_events": len(child.session)},
        )

    return tools.register(ToolDefinition(
        name=name,
        description="Delegate a self-contained task to a fresh subagent and "
                    "return its final answer.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "model": {"type": "string"},
            },
            "required": ["task"],
        },
        execute=delegate,
    ))


__all__ = ["register_subagent_tool"]
