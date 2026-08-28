"""Folds a raw ``StreamChunk`` sequence into one assistant message."""

from __future__ import annotations

from .types import (
    AssistantProvenance,
    ContentBlock,
    Message,
    MessageSource,
    MessageStop,
    ReasoningBlock,
    ReasoningDelta,
    StreamChunk,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)


class StreamAssembler:
    """Shared assembler: chunks in, one ``Message`` plus usage out.

    Text and reasoning deltas accumulate into their current block; a change
    of block kind opens a new block so ordering is preserved. Tool calls are
    assembled by id from start/delta/end chunks.
    """

    def __init__(self, provenance: AssistantProvenance) -> None:
        self._provenance = provenance
        self._blocks: list[ContentBlock] = []
        self._open_tool_calls: dict[str, dict[str, str]] = {}
        self._tool_call_order: list[str] = []
        self._usage: TokenUsage | None = None
        self._stopped = False

    @property
    def usage(self) -> TokenUsage | None:
        """Token usage from the terminal ``MessageStop``, when reported."""
        return self._usage

    def feed(self, chunk: StreamChunk) -> None:
        """Consume one chunk."""
        if isinstance(chunk, TextDelta):
            self._append_text(chunk.text, "text")
        elif isinstance(chunk, ReasoningDelta):
            self._append_text(chunk.thinking, "reasoning")
        elif isinstance(chunk, ToolCallStart):
            self._open_tool_calls[chunk.id] = {"name": chunk.name, "arguments": ""}
            self._tool_call_order.append(chunk.id)
        elif isinstance(chunk, ToolCallDelta):
            self._open_tool_calls[chunk.id]["arguments"] += chunk.arguments_delta
        elif isinstance(chunk, ToolCallEnd):
            call = self._open_tool_calls.pop(chunk.id)
            self._blocks.append(
                ToolCallBlock(id=chunk.id, name=call["name"], arguments=call["arguments"])
            )
        elif isinstance(chunk, MessageStop):
            self._usage = chunk.usage
            self._stopped = True
        else:  # pragma: no cover - unreachable for the closed union
            raise TypeError(f"unknown stream chunk: {chunk!r}")

    def _append_text(self, text: str, kind: str) -> None:
        if not text:
            return
        last = self._blocks[-1] if self._blocks else None
        if kind == "text" and isinstance(last, TextBlock):
            self._blocks[-1] = TextBlock(text=last.text + text)
        elif kind == "reasoning" and isinstance(last, ReasoningBlock):
            self._blocks[-1] = ReasoningBlock(thinking=last.thinking + text)
        elif kind == "text":
            self._blocks.append(TextBlock(text=text))
        else:
            self._blocks.append(ReasoningBlock(thinking=text))

    def finish(self, *, interrupted: bool = False) -> Message:
        """Assemble the assistant message from everything fed so far.

        ``interrupted`` finalizes the delivered prefix of a cancelled stream;
        tool calls that never closed are dropped.
        """
        if not interrupted and (self._open_tool_calls or not self._stopped):
            raise ValueError("stream not finished: missing tool-call-end or message-stop")
        return Message(
            role="assistant",
            content=tuple(self._blocks),
            source=MessageSource(kind="model", provenance=self._provenance),
        )
