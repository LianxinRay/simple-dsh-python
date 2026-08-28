"""JSONL durability for the session log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from .events import SessionEvent, event_from_json, event_to_json


class JsonlSink:
    """Appends every session event as one JSON line.

    Attach with ``session.add_sink(sink)``. The canonical log is stored
    verbatim, so reopening the file replays identical history.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self.path.open("a", encoding="utf-8")

    def __call__(self, event: SessionEvent) -> None:
        self._file.write(json.dumps(event_to_json(event), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()


def load_jsonl(path: str | Path) -> list[SessionEvent]:
    """Read a JSONL log back into events, in stored order."""
    events = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                events.append(event_from_json(json.loads(line)))
    return events
