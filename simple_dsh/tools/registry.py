"""The scoped tool registry and its guarded execution pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from ..cordis import Context, Disposer
from ..llm import ContentBlock, TextBlock, ToolCallBlock

ToolBody = Callable[[dict[str, Any]], Awaitable["ToolResult"]]
Guard = Callable[[ToolCallBlock], Literal["allow", "deny"] | Awaitable[Literal["allow", "deny"]]]


@dataclass
class ToolResult:
    """A tool call's model-facing outcome."""

    content: list[ContentBlock] = field(default_factory=list)
    is_error: bool = False
    meta: dict[str, Any] | None = None  # tool-private, MUST be JSON-serializable

    @classmethod
    def text(cls, text: str, *, is_error: bool = False) -> "ToolResult":
        """Build a plain-text result."""
        return cls(content=[TextBlock(text=text)], is_error=is_error)


@dataclass(frozen=True)
class ToolDefinition:
    """A model-facing tool: schema plus async body."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    execute: ToolBody

    def schema(self) -> dict[str, Any]:
        """The tool schema that joins prompt assembly."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class _CallState:
    """The mutable decision object flowing through the pre-execute waterfall."""

    call: ToolCallBlock
    action: Literal["allow", "deny"] = "allow"
    reason: str = ""


class ToolRegistry:
    """The ``ctx.tools`` service.

    Holds the scoped registry and runs the guarded pipeline. Pipeline policy
    lives on context events so hooks and guards attach without importing the
    loop.
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._tools: dict[str, ToolDefinition] = {}
        self._guards: list[tuple[Guard, str]] = []

    # ------------------------------------------------------------- registry

    def register(self, definition: ToolDefinition) -> Disposer:
        """Register a tool; returns the disposer removing it."""
        self._tools[definition.name] = definition

        def dispose() -> None:
            self._tools.pop(definition.name, None)

        return dispose

    def get(self, name: str) -> ToolDefinition | None:
        """Return the tool registered under ``name``, if any."""
        return self._tools.get(name)

    def schemas(self) -> tuple[dict[str, Any], ...]:
        """All registered tool schemas, for request assembly."""
        return tuple(tool.schema() for tool in self._tools.values())

    # ---------------------------------------------------------------- guards

    def register_guard(self, guard: Guard, *, name: str = "guard") -> Disposer:
        """Register a monotonic guard: it may deny or abstain, never re-allow."""
        entry = (guard, name)
        self._guards.append(entry)

        def dispose() -> None:
            if entry in self._guards:
                self._guards.remove(entry)

        return dispose

    # -------------------------------------------------------------- pipeline

    async def execute(self, call: ToolCallBlock) -> ToolResult:
        """Run one tool call through the full guarded pipeline."""
        try:
            state = _CallState(call=call)
            state = await self._ctx.waterfall("tools/pre-execute", state)
            if state.action == "deny":
                result = ToolResult.text(f"denied: {state.reason or 'by policy'}", is_error=True)
            else:
                denial = await self._run_guards(call)
                if denial is not None:
                    result = denial
                else:
                    result = await self._dispatch(call)
            result = await self._ctx.waterfall("tools/post-execute", result)
            return result
        except Exception as exc:  # pipeline normalization: never raise
            return ToolResult.text(f"{type(exc).__name__}: {exc}", is_error=True)

    async def _run_guards(self, call: ToolCallBlock) -> ToolResult | None:
        from ..cordis import maybe_await

        for guard, name in self._guards:
            verdict = await maybe_await(guard(call))
            if verdict == "deny":
                return ToolResult.text(f"denied: guard {name}", is_error=True)
        return None

    async def _dispatch(self, call: ToolCallBlock) -> ToolResult:
        definition = self._tools.get(call.name)
        if definition is None:
            return ToolResult.text(f"unknown tool: {call.name}", is_error=True)
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError as exc:
            return ToolResult.text(f"invalid tool arguments JSON: {exc}", is_error=True)
        if not isinstance(args, dict):
            return ToolResult.text("tool arguments must be a JSON object", is_error=True)

        async def body(_: Any) -> ToolResult:
            return await definition.execute(args)

        return await self._ctx.waterfall_terminal("tools/execute", call, body)
