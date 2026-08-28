"""Approval interaction (P2).

Ported from ``packages/interaction``: ``ctx.approval`` answers one-shot
confirmation prompts. A ``tools/pre-execute`` listener asks before running
configured tools; when approval is absent, unanswerable, or refused, the
call is denied — never silently allowed.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

from .cordis import Context, Disposer, maybe_await
from .llm import ToolCallBlock

# A responder answers one approval prompt: True allows, anything else denies.
Responder = Callable[[ToolCallBlock], bool | Awaitable[bool]]


class ApprovalService:
    """The ``ctx.approval`` service. Without a responder, every ask denies."""

    def __init__(self, responder: Responder | None = None) -> None:
        self._responder = responder

    async def ask(self, call: ToolCallBlock) -> bool:
        """Ask once. Absent or failing responder → deny (fail closed)."""
        if self._responder is None:
            return False
        try:
            return bool(await maybe_await(self._responder(call)))
        except Exception:
            return False


def console_responder(call: ToolCallBlock) -> bool:
    """Terminal responder: prompt y/N on stdin."""
    answer = input(f"approve tool call {call.name}({call.arguments[:200]})? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def require_approval(
    ctx: Context, approval: ApprovalService, tool_names: Iterable[str]
) -> Disposer:
    """Deny calls to ``tool_names`` unless the approval service allows them."""
    gated = frozenset(tool_names)

    async def listener(state, next_):
        if state.action == "allow" and state.call.name in gated:
            if not await approval.ask(state.call):
                state.action = "deny"
                state.reason = "approval refused or unavailable"
                return state
        return await next_(state)

    return ctx.on("tools/pre-execute", listener)


__all__ = ["ApprovalService", "console_responder", "require_approval"]
