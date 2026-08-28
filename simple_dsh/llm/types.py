"""The conversation and streaming vocabulary every layer shares.

Ported from ``packages/llm/llm``: a conversation is ``Message``\\ s; a message
is an array of typed content blocks; adapters stream raw ``StreamChunk``\\ s
that the assembler folds into one assistant message. Blocks, chunks, and
messages are lossless JSON so the durable session log can store them
verbatim.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, Union, runtime_checkable

Role = Literal["system", "user", "assistant"]

_message_ids = itertools.count(1)


def next_message_id() -> str:
    """Return a process-unique message id."""
    return f"msg-{next(_message_ids)}"


# ---------------------------------------------------------------- blocks


@dataclass(frozen=True)
class TextBlock:
    """Visible assistant or user text."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ReasoningBlock:
    """Model thinking, distinct from visible text."""

    thinking: str
    type: Literal["reasoning"] = "reasoning"


@dataclass(frozen=True)
class ImageBlock:
    """An image by URL or base64 data."""

    url: str | None = None
    data: str | None = None
    media_type: str | None = None
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class ToolCallBlock:
    """A model-requested tool invocation; ``arguments`` is raw JSON."""

    id: str
    name: str
    arguments: str
    type: Literal["tool-call"] = "tool-call"


@dataclass(frozen=True)
class ToolResultBlock:
    """One completed tool call's model-facing outcome."""

    tool_call_id: str
    content: tuple["ContentBlock", ...] = ()
    is_error: bool = False
    type: Literal["tool-result"] = "tool-result"


ContentBlock = Union[TextBlock, ReasoningBlock, ImageBlock, ToolCallBlock, ToolResultBlock]

_BLOCK_TYPES: dict[str, type] = {
    "text": TextBlock,
    "reasoning": ReasoningBlock,
    "image": ImageBlock,
    "tool-call": ToolCallBlock,
    "tool-result": ToolResultBlock,
}


def block_to_json(block: ContentBlock) -> dict[str, Any]:
    """Serialize a content block to lossless JSON."""
    data: dict[str, Any] = {"type": block.type}
    if isinstance(block, TextBlock):
        data["text"] = block.text
    elif isinstance(block, ReasoningBlock):
        data["thinking"] = block.thinking
    elif isinstance(block, ImageBlock):
        data.update({"url": block.url, "data": block.data, "media_type": block.media_type})
    elif isinstance(block, ToolCallBlock):
        data.update({"id": block.id, "name": block.name, "arguments": block.arguments})
    elif isinstance(block, ToolResultBlock):
        data.update(
            {
                "tool_call_id": block.tool_call_id,
                "content": [block_to_json(b) for b in block.content],
                "is_error": block.is_error,
            }
        )
    else:  # pragma: no cover - unreachable for the closed union
        raise TypeError(f"unknown content block: {block!r}")
    return data


def block_from_json(data: dict[str, Any]) -> ContentBlock:
    """Deserialize a content block written by :func:`block_to_json`."""
    kind = data["type"]
    if kind not in _BLOCK_TYPES:
        raise ValueError(f"unknown content block type: {kind!r}")
    if kind == "tool-result":
        return ToolResultBlock(
            tool_call_id=data["tool_call_id"],
            content=tuple(block_from_json(b) for b in data.get("content", [])),
            is_error=data.get("is_error", False),
        )
    payload = {k: v for k, v in data.items() if k != "type"}
    return _BLOCK_TYPES[kind](**payload)


# --------------------------------------------------------------- messages


@dataclass(frozen=True)
class AssistantProvenance:
    """Provider/model identity of an assistant message."""

    provider: str
    model: str


@dataclass(frozen=True)
class MessageSource:
    """Where a message came from. ``kind``: user | plugin | model | tool."""

    kind: str
    provenance: AssistantProvenance | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One immutable message shared by delivery, history, and requests."""

    role: Role
    content: tuple[ContentBlock, ...]
    source: MessageSource
    id: str = field(default_factory=next_message_id)


def message_to_json(message: Message) -> dict[str, Any]:
    """Serialize a message to lossless JSON."""
    source: dict[str, Any] = {"kind": message.source.kind, **message.source.extra}
    if message.source.provenance is not None:
        source["provenance"] = {
            "provider": message.source.provenance.provider,
            "model": message.source.provenance.model,
        }
    return {
        "id": message.id,
        "role": message.role,
        "content": [block_to_json(b) for b in message.content],
        "source": source,
    }


def message_from_json(data: dict[str, Any]) -> Message:
    """Deserialize a message written by :func:`message_to_json`."""
    source_data = dict(data["source"])
    provenance_data = source_data.pop("provenance", None)
    kind = source_data.pop("kind")
    provenance = AssistantProvenance(**provenance_data) if provenance_data else None
    return Message(
        id=data["id"],
        role=data["role"],
        content=tuple(block_from_json(b) for b in data["content"]),
        source=MessageSource(kind=kind, provenance=provenance, extra=source_data),
    )


# ------------------------------------------------------------------ chunks


@dataclass(frozen=True)
class TextDelta:
    """A fragment of visible assistant text."""

    text: str
    type: Literal["text-delta"] = "text-delta"


@dataclass(frozen=True)
class ReasoningDelta:
    """A fragment of assistant thinking."""

    thinking: str
    type: Literal["reasoning-delta"] = "reasoning-delta"


@dataclass(frozen=True)
class ToolCallStart:
    """Opens a streamed tool call."""

    id: str
    name: str
    type: Literal["tool-call-start"] = "tool-call-start"


@dataclass(frozen=True)
class ToolCallDelta:
    """Appends raw JSON to a streamed tool call's arguments."""

    id: str
    arguments_delta: str
    type: Literal["tool-call-delta"] = "tool-call-delta"


@dataclass(frozen=True)
class ToolCallEnd:
    """Closes a streamed tool call."""

    id: str
    type: Literal["tool-call-end"] = "tool-call-end"


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting reported by an adapter."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class MessageStop:
    """Ends the streamed assistant message, optionally with usage."""

    usage: TokenUsage | None = None
    type: Literal["message-stop"] = "message-stop"


StreamChunk = Union[TextDelta, ReasoningDelta, ToolCallStart, ToolCallDelta, ToolCallEnd, MessageStop]

_CHUNK_TYPES: dict[str, type] = {
    "text-delta": TextDelta,
    "reasoning-delta": ReasoningDelta,
    "tool-call-start": ToolCallStart,
    "tool-call-delta": ToolCallDelta,
    "tool-call-end": ToolCallEnd,
    "message-stop": MessageStop,
}


def chunk_to_json(chunk: StreamChunk) -> dict[str, Any]:
    """Serialize a stream chunk to lossless JSON."""
    data: dict[str, Any] = {"type": chunk.type}
    if isinstance(chunk, TextDelta):
        data["text"] = chunk.text
    elif isinstance(chunk, ReasoningDelta):
        data["thinking"] = chunk.thinking
    elif isinstance(chunk, ToolCallStart):
        data.update({"id": chunk.id, "name": chunk.name})
    elif isinstance(chunk, ToolCallDelta):
        data.update({"id": chunk.id, "arguments_delta": chunk.arguments_delta})
    elif isinstance(chunk, ToolCallEnd):
        data["id"] = chunk.id
    elif isinstance(chunk, MessageStop):
        if chunk.usage is not None:
            data["usage"] = {
                "input_tokens": chunk.usage.input_tokens,
                "output_tokens": chunk.usage.output_tokens,
            }
    else:  # pragma: no cover - unreachable for the closed union
        raise TypeError(f"unknown stream chunk: {chunk!r}")
    return data


def chunk_from_json(data: dict[str, Any]) -> StreamChunk:
    """Deserialize a stream chunk written by :func:`chunk_to_json`."""
    kind = data["type"]
    if kind not in _CHUNK_TYPES:
        raise ValueError(f"unknown stream chunk type: {kind!r}")
    if kind == "message-stop":
        usage_data = data.get("usage")
        return MessageStop(usage=TokenUsage(**usage_data) if usage_data else None)
    payload = {k: v for k, v in data.items() if k != "type"}
    return _CHUNK_TYPES[kind](**payload)


# ----------------------------------------------------------------- request


@dataclass(frozen=True)
class ModelRequest:
    """The fully assembled model request for one step."""

    model: str
    messages: tuple[Message, ...]
    system: str = ""
    tools: tuple[dict[str, Any], ...] = ()


# ----------------------------------------------------------------- adapter


@runtime_checkable
class LlmAdapter(Protocol):
    """The seam every model provider implements."""

    provider: str
    models: tuple[str, ...]

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Stream the model response as raw chunks."""
        ...


class LlmRegistry:
    """The ``ctx.llm`` service: adapter registration and model resolution."""

    def __init__(self) -> None:
        self._adapters: list[LlmAdapter] = []

    def register_adapter(self, adapter: LlmAdapter) -> Callable[[], None]:
        """Register an adapter; returns the disposer removing it."""
        self._adapters.append(adapter)

        def dispose() -> None:
            if adapter in self._adapters:
                self._adapters.remove(adapter)

        return dispose

    def resolve(self, model: str) -> LlmAdapter:
        """Return the adapter serving ``model``; fail loud when none does."""
        for adapter in self._adapters:
            if model in adapter.models:
                return adapter
        raise KeyError(f"no LLM adapter serves model {model!r}")
