"""The in-memory, event-sourced session model."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from ..llm import (
    ContentBlock,
    Message,
    MessageSource,
    TextBlock,
    ToolResultBlock,
    message_from_json,
    message_to_json,
)
from .events import SessionEvent, is_json_value

# A projection turns an event payload into zero or more model-visible
# messages. New event types join derived history by registering one.
Projection = Callable[[dict[str, Any]], Iterable[Message]]


class Session:
    """An append-only log of session events — the single source of truth.

    Every appended payload must be lossless JSON (enforced at append, like
    ``Session.append``'s ``isJsonValue`` validation upstream). Sequence
    numbers are contiguous. Model-visible history is derived by replaying
    registered projections over the log; ``assistant/chunk`` events carry
    token-level replay fidelity but never enter derived history.
    """

    def __init__(self, events: Iterable[SessionEvent] = ()) -> None:
        self._events: list[SessionEvent] = []
        self._projections: dict[str, Projection] = {}
        self._sinks: list[Callable[[SessionEvent], None]] = []
        self.register_projection("user/message", _project_user_message)
        self.register_projection("assistant/message", _project_assistant_message)
        self.register_projection("tool/result", _project_tool_result)
        for event in events:
            if event.seq != len(self._events):
                raise ValueError("seeded events must have contiguous seq from 0")
            self._events.append(event)

    # ------------------------------------------------------------- appending

    def register_projection(self, event_type: str, projection: Projection) -> None:
        """Teach derived history how to render ``event_type`` payloads."""
        self._projections[event_type] = projection

    def add_sink(self, sink: Callable[[SessionEvent], None]) -> None:
        """Attach a durable sink (e.g. JSONL) that receives every event."""
        self._sinks.append(sink)

    def append(self, event_type: str, data: dict[str, Any]) -> SessionEvent:
        """Append one event; rejects non-JSON payloads at the source."""
        if not is_json_value(data):
            raise TypeError(f"session event {event_type!r} payload is not lossless JSON")
        event = SessionEvent(seq=len(self._events), type=event_type, data=data)
        self._events.append(event)
        for sink in self._sinks:
            sink(event)
        return event

    # -------------------------------------------------------------- reading

    def __iter__(self) -> Iterator[SessionEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def at(self, seq: int) -> SessionEvent:
        """Return the event with sequence number ``seq``."""
        return self._events[seq]

    def derive_messages(self) -> list[Message]:
        """Project model-visible history from the log, in event order.

        A ``compaction/summary`` event is a boundary: derived history becomes
        the summary message, then any verbatim ``kept`` messages carried by
        the event, then projections of events after it.
        """
        start = 0
        boundary: dict[str, Any] | None = None
        for index, event in enumerate(self._events):
            if event.type == "compaction/summary":
                start = index + 1
                boundary = event.data
        messages: list[Message] = []
        if boundary is not None:
            messages.append(message_from_json(boundary["message"]))
            messages.extend(message_from_json(m) for m in boundary.get("kept", []))
        for event in self._events[start:]:
            projection = self._projections.get(event.type)
            if projection is not None:
                messages.extend(projection(event.data))
        return messages


# ------------------------------------------------------------- projections


def _project_user_message(data: dict[str, Any]) -> Iterable[Message]:
    yield message_from_json(data)


def _project_assistant_message(data: dict[str, Any]) -> Iterable[Message]:
    yield message_from_json(data["message"])


def _project_tool_result(data: dict[str, Any]) -> Iterable[Message]:
    yield message_from_json(data["message"])


def make_tool_result_message(
    tool_call_id: str, content: list[ContentBlock], is_error: bool
) -> Message:
    """Build the user-role message that carries a tool result into history."""
    return Message(
        role="user",
        content=(
            ToolResultBlock(tool_call_id=tool_call_id, content=tuple(content), is_error=is_error),
        ),
        source=MessageSource(kind="tool"),
    )


def make_user_message(text: str, *, source_kind: str = "user", **extra: Any) -> Message:
    """Build a user-role message; ``source.kind`` tells producers apart."""
    return Message(
        role="user",
        content=(TextBlock(text=text),),
        source=MessageSource(kind=source_kind, extra=extra),
    )


__all__ = ["Session", "make_tool_result_message", "make_user_message", "message_to_json"]
