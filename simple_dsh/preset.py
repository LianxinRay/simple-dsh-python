"""Preset composition (P2).

Ported from ``packages/preset``: a preset is a per-session composition
document — which tools, which guards, which model. Presets are declared as
JSON files (JSON is valid YAML 1.2, so a YAML loader can replace the parser
later without changing the format). Missing referents fail loud.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PRESET: dict[str, Any] = {
    "model": "deepseek-chat",
    "tools": {
        "fs": True,
        "shell": True,
        "web": False,
        "todo": True,
        "subagent": True,
    },
    "approval": {"tools": []},
    "guards": {"timeout": 120.0, "repeat_call_limit": 3},
    "compaction": {"enabled": True, "max_tokens": 8000, "keep_recent": 4},
}
"""The composition every session gets unless its preset says otherwise."""


def load_preset(path: str | Path) -> dict[str, Any]:
    """Load a preset JSON file, deep-merged over :data:`DEFAULT_PRESET`."""
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    return merge_preset(user)


def merge_preset(user: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a partial preset over the defaults."""
    merged: dict[str, Any] = {**DEFAULT_PRESET, **user}
    for section in ("tools", "approval", "guards", "compaction"):
        merged[section] = {**DEFAULT_PRESET[section], **user.get(section, {})}
    return merged


__all__ = ["DEFAULT_PRESET", "load_preset", "merge_preset"]
