"""Context compaction (P2).

Ported from ``packages/compaction``: when derived history grows past a token
budget, the older span is summarized and the summary is appended to the log
as a ``compaction/summary`` event. History stays reconstructable — the raw
log is never rewritten; derivation just starts from the newest summary.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from .llm import (
    LlmRegistry,
    Message,
    ModelRequest,
    StreamAssembler,
    AssistantProvenance,
    TextBlock,
)
from .session import Session, make_user_message

Summarizer = Callable[[list[Message]], Awaitable[str]]
"""Turns a span of messages into summary text."""


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate: 4 chars ≈ 1 token, over all text-ish content."""
    total = 0
    for message in messages:
        for block in message.content:
            total += len(getattr(block, "text", "") or getattr(block, "thinking", ""))
            total += len(getattr(block, "arguments", ""))
    return total // 4


class LlmSummarizer:
    """Summarizes with the session's own model via the adapter seam."""

    def __init__(self, llm: LlmRegistry, model: str) -> None:
        self._llm = llm
        self._model = model

    async def __call__(self, messages: list[Message]) -> str:
        adapter = self._llm.resolve(self._model)
        ask = make_user_message(
            "Summarize the conversation so far in under 200 words, preserving "
            "decisions, file changes, and outstanding tasks."
        )
        request = ModelRequest(
            model=self._model,
            messages=tuple(messages) + (ask,),
            system="You are a conversation summarizer. Output only the summary.",
        )
        assembler = StreamAssembler(AssistantProvenance(adapter.provider, self._model))
        async for chunk in adapter.stream(request):
            assembler.feed(chunk)
        summary = assembler.finish()
        return "".join(b.text for b in summary.content if isinstance(b, TextBlock))


class CompactionService:
    """The ``ctx.compaction`` service: budget check + log-backed compact."""

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        max_tokens: int = 8000,
        keep_recent: int = 4,
    ) -> None:
        self._summarizer = summarizer
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

    async def maybe_compact(self, session: Session) -> bool:
        """Compact ``session``'s older history when over budget.

        Appends ``compaction/start`` → ``compaction/summary`` →
        ``compaction/end``. Returns True when a compaction happened.
        """
        messages = session.derive_messages()
        if estimate_tokens(messages) <= self.max_tokens:
            return False
        if len(messages) <= self.keep_recent + 1:
            return False  # nothing meaningful to fold away
        covered = messages[: -self.keep_recent]
        kept = messages[-self.keep_recent:]
        session.append("compaction/start", {"covered_messages": len(covered)})
        text = await self._summarizer(covered)
        summary = make_user_message(
            f"[conversation summary]\n{text}", source_kind="plugin", form="compaction"
        )
        from .llm import message_to_json

        session.append(
            "compaction/summary",
            {
                "message": message_to_json(summary),
                "kept": [message_to_json(m) for m in kept],
            },
        )
        session.append("compaction/end", {})
        return True


__all__ = ["CompactionService", "LlmSummarizer", "estimate_tokens"]
