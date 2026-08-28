"""System-prompt assembly (P0-4).

Ported from ``packages/core/system-prompt``: plugins register named prompt
sections; before each request the service renders them in priority order and
joins them into the system text. Tool schemas are collected separately from
``ctx.tools`` at request time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .cordis import Disposer, maybe_await

SectionRender = Callable[[], str | Awaitable[str]]


@dataclass(frozen=True)
class PromptSection:
    """One named prompt section. Lower ``priority`` renders earlier."""

    id: str
    render: SectionRender
    priority: int = 100
    extra: dict[str, Any] = field(default_factory=dict)


class SystemPrompt:
    """The ``ctx.systemPrompt`` service: section registry + assembly."""

    def __init__(self) -> None:
        self._sections: dict[str, PromptSection] = {}

    def register_section(self, section: PromptSection) -> Disposer:
        """Register a section; returns the disposer removing it."""
        self._sections[section.id] = section

        def dispose() -> None:
            self._sections.pop(section.id, None)

        return dispose

    async def assemble(self) -> str:
        """Render all sections in priority order and join with blank lines."""
        parts = []
        for section in sorted(self._sections.values(), key=lambda s: (s.priority, s.id)):
            text = await maybe_await(section.render())
            if text:
                parts.append(text)
        return "\n\n".join(parts)
