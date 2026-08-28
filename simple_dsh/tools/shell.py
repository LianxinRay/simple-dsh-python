"""Shell execution tool (P1).

A minimal ``packages/shell`` analogue: one model-facing ``bash`` tool running
commands through the platform shell in a working directory. Output is
truncated; a non-zero exit code yields an ``is_error`` result. Deadlines
belong to the guard layer (``simple_dsh.guard``), not the tool body.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..cordis import Disposer
from ..llm import TextBlock
from .registry import ToolDefinition, ToolRegistry, ToolResult

_MAX_OUTPUT = 30_000


def register_shell_tool(
    tools: ToolRegistry, workdir: str | Path, *, name: str = "bash"
) -> Disposer:
    """Register the shell tool rooted at ``workdir``; returns its disposer."""
    root = Path(workdir).resolve()

    async def run(args):
        command = args["command"]
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        output = stdout.decode("utf-8", errors="replace")
        truncated = len(output) > _MAX_OUTPUT
        if truncated:
            output = output[:_MAX_OUTPUT] + "\n... [truncated]"
        code = process.returncode
        suffix = f"\n[exit code: {code}]" if code else ""
        return ToolResult(
            content=[TextBlock(text=(output or "(no output)") + suffix)],
            is_error=code != 0,
            meta={"exit_code": code, "truncated": truncated},
        )

    return tools.register(ToolDefinition(
        name=name,
        description=f"Run a shell command in {root}. Returns combined stdout/stderr.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        execute=run,
    ))


__all__ = ["register_shell_tool"]
