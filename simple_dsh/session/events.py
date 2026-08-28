"""Session event vocabulary and JSON guarantees.

The core event types mirror ``SessionEventMap``. Plugins may append their own
event types (the Python analogue of declaration merging) and register
projections for them on :class:`~simple_dsh.session.session.Session`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

TurnEndReason = Literal["completed", "rejected", "empty", "cancelled", "error"]

CORE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "turn/start",
        "turn/end",
        "step/start",
        "step/end",
        "user/message",
        "assistant/chunk",
        "assistant/message",
        "tool/call",
        "tool/result",
        "request/header",
    }
)
"""The append-only core event vocabulary."""


def is_json_value(value: Any) -> bool:
    """Return True when ``value`` round-trips through JSON losslessly."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_value(v) for k, v in value.items())
    return False


@dataclass(frozen=True)
class SessionEvent:
    """One entry of the log: a type tag and a lossless-JSON payload.

    ``seq`` numbers are contiguous and monotonic; ``time`` is a Unix
    timestamp assigned at append.
    """

    seq: int
    type: str
    data: dict[str, Any]
    time: float = field(default_factory=time.time)


def event_to_json(event: SessionEvent) -> dict[str, Any]:
    """Serialize an event to one JSONL record."""
    return {"seq": event.seq, "time": event.time, "type": event.type, "data": event.data}


def event_from_json(data: dict[str, Any]) -> SessionEvent:
    """Deserialize an event written by :func:`event_to_json`."""
    return SessionEvent(seq=data["seq"], time=data["time"], type=data["type"], data=data["data"])
