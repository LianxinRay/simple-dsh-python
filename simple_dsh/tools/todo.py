"""Todo tool (P2).

Ported from ``packages/todo``: the model-facing ``todo_write`` tool. State is
log-backed — every write appends a whole-list ``todo/write`` snapshot to the
session log; latest write wins on replay. Never part of derived history.
"""

from __future__ import annotations

from ..cordis import Disposer
from ..session import Session
from .registry import ToolDefinition, ToolRegistry, ToolResult

_VALID_STATUS = {"pending", "in_progress", "done"}


def register_todo_tool(tools: ToolRegistry, session: Session) -> Disposer:
    """Register ``todo_write``; returns its disposer."""

    async def todo_write(args):
        todos = []
        for item in args["todos"]:
            status = item.get("status", "pending")
            if status not in _VALID_STATUS:
                return ToolResult.text(
                    f"invalid status {status!r}; expected one of {sorted(_VALID_STATUS)}",
                    is_error=True,
                )
            todos.append({"title": item["title"], "status": status})
        session.append("todo/write", {"todos": todos})
        lines = [f"[{t['status']}] {t['title']}" for t in todos]
        return ToolResult.text("todo list updated:\n" + "\n".join(lines))

    return tools.register(ToolDefinition(
        name="todo_write",
        description="Replace the task list. Each item: title + status "
                    "(pending | in_progress | done).",
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["title", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        execute=todo_write,
    ))


def latest_todos(session: Session) -> list[dict]:
    """Replay the log for the newest todo snapshot (latest write wins)."""
    todos: list[dict] = []
    for event in session:
        if event.type == "todo/write":
            todos = event.data["todos"]
    return todos


__all__ = ["latest_todos", "register_todo_tool"]
