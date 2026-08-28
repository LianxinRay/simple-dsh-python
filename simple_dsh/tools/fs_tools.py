"""Filesystem tools (P1), confined to a workspace root.

A minimal ``packages/fs`` analogue: ``read_file`` / ``write_file`` /
``edit_file`` / ``list_directory`` as model-facing tools. All paths resolve
against the workspace root; escapes fail loud — the root is the policy seam.
"""

from __future__ import annotations

from pathlib import Path

from ..cordis import Disposer
from .registry import ToolDefinition, ToolRegistry, ToolResult


class WorkspacePolicy:
    """Resolves model-supplied paths and refuses escapes from the root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` under the root; escapes raise ``ValueError``."""
        resolved = (self.root / path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"path {path!r} escapes the workspace root")
        return resolved


def register_fs_tools(tools: ToolRegistry, root: str | Path) -> list[Disposer]:
    """Register the four filesystem tools; returns their disposers."""
    policy = WorkspacePolicy(root)

    async def read_file(args):
        target = policy.resolve(args["path"])
        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()
        offset = max(int(args.get("offset", 1)) - 1, 0)
        limit = args.get("limit")
        selected = lines[offset:] if limit is None else lines[offset : offset + int(limit)]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        return ToolResult.text("\n".join(numbered))

    async def write_file(args):
        target = policy.resolve(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return ToolResult.text(f"wrote {target}")

    async def edit_file(args):
        target = policy.resolve(args["path"])
        text = target.read_text(encoding="utf-8")
        old, new = args["old_string"], args["new_string"]
        count = text.count(old)
        if count == 0:
            return ToolResult.text("old_string not found", is_error=True)
        if count > 1:
            return ToolResult.text(f"old_string is not unique ({count} matches)", is_error=True)
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult.text(f"edited {target}")

    async def list_directory(args):
        target = policy.resolve(args.get("path", "."))
        entries = sorted(
            ("/" if entry.is_dir() else "") + entry.name
            for entry in target.iterdir()
        )
        return ToolResult.text("\n".join(entries))

    text_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    return [
        tools.register(ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file with line numbers. Optional 1-based offset and limit.",
            parameters={
                **text_schema,
                "properties": {
                    **text_schema["properties"],
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
            execute=read_file,
        )),
        tools.register(ToolDefinition(
            name="write_file",
            description="Create or completely overwrite a UTF-8 text file.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            execute=write_file,
        )),
        tools.register(ToolDefinition(
            name="edit_file",
            description="Replace a unique exact string in a file with a new string.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            execute=edit_file,
        )),
        tools.register(ToolDefinition(
            name="list_directory",
            description="List a directory's entries; directories carry a trailing slash.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            execute=list_directory,
        )),
    ]


__all__ = ["WorkspacePolicy", "register_fs_tools"]
