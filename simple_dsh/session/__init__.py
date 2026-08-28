"""The append-only session event log (P0-3).

Ported from ``packages/core/session``: a ``Session`` is an append-only log of
typed session events — the single source of truth for an agent's whole
interaction history. The model-visible message history is *derived* from the
log, never stored separately; replay is re-derivation from the same events.
A JSONL sink makes the log durable.
"""

from .events import (
    CORE_EVENT_TYPES,
    SessionEvent,
    TurnEndReason,
    event_from_json,
    event_to_json,
    is_json_value,
)
from .persistence import JsonlSink, load_jsonl
from .session import Session, make_tool_result_message, make_user_message

__all__ = [
    "CORE_EVENT_TYPES",
    "JsonlSink",
    "Session",
    "SessionEvent",
    "TurnEndReason",
    "event_from_json",
    "event_to_json",
    "is_json_value",
    "load_jsonl",
    "make_tool_result_message",
    "make_user_message",
]
