"""Loop-hygiene guards (P1).

Ported from ``packages/guard``:

- :func:`register_timeout` — a ``tools/execute`` around-listener enforcing a
  per-call deadline. Timeouts surface as normalized ``is_error`` results via
  the registry's pipeline normalization, so the loop never crashes.
- :class:`RepeatCallGuard` — an advisory monotonic guard denying a tool call
  that repeats identically (same name and arguments) more than ``limit``
  times in a row, breaking degenerate retry loops.
"""

from __future__ import annotations

import asyncio

from .cordis import Context, Disposer
from .llm import ToolCallBlock
from .tools import ToolRegistry


def register_timeout(ctx: Context, seconds: float) -> Disposer:
    """Enforce a per-tool-call deadline via the ``tools/execute`` waterfall."""

    async def around(call, next_):
        return await asyncio.wait_for(next_(), timeout=seconds)

    return ctx.on("tools/execute", around)


class RepeatCallGuard:
    """Denies the ``limit``\\ +1-th consecutive identical tool call."""

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self._last: tuple[str, str] | None = None
        self._streak = 0

    def __call__(self, call: ToolCallBlock) -> str:
        key = (call.name, call.arguments)
        if key == self._last:
            self._streak += 1
        else:
            self._last, self._streak = key, 1
        return "deny" if self._streak > self.limit else "allow"

    def register(self, tools: ToolRegistry) -> Disposer:
        """Attach this guard to a registry; returns its disposer."""
        return tools.register_guard(self, name="repeat-call")


__all____all__ = ["RepeatCallGuard", "register_timeout"]
