"""SQLite durability for the session log (P2).

Ported from the SQLite backend in ``packages/session``: a monotonic
``SCHEMA_VERSION`` with no compatibility promise pre-release; events stored
verbatim as JSON so replay reproduces the canonical log.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .events import SessionEvent, event_from_json, event_to_json

SCHEMA_VERSION = 1


class SqliteSink:
    """Appends every session event into a SQLite table. Attach via
    ``session.add_sink(sink)``. Thread-safe enough for one writer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "  seq INTEGER PRIMARY KEY,"
            "  time REAL NOT NULL,"
            "  type TEXT NOT NULL,"
            "  data TEXT NOT NULL"
            ")"
        )
        version = self._db.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if version is None:
            self._db.execute(
                "INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
        elif int(version[0]) != SCHEMA_VERSION:
            self._db.close()
            raise ValueError(
                f"schema version {version[0]} != {SCHEMA_VERSION}; "
                "pre-release backends reject old on-disk formats"
            )
        self._db.commit()

    def __call__(self, event: SessionEvent) -> None:
        record = event_to_json(event)
        self._db.execute(
            "INSERT INTO events (seq, time, type, data) VALUES (?, ?, ?, ?)",
            (record["seq"], record["time"], record["type"], json.dumps(record["data"], ensure_ascii=False)),
        )
        self._db.commit()

    def close(self) -> None:
        """Close the database handle."""
        self._db.close()


def load_sqlite(path: str | Path) -> list[SessionEvent]:
    """Read all events back in seq order."""
    db = sqlite3.connect(str(path))
    try:
        rows = db.execute(
            "SELECT seq, time, type, data FROM events ORDER BY seq"
        ).fetchall()
    finally:
        db.close()
    return [
        event_from_json({"seq": seq, "time": time, "type": type_, "data": json.loads(data)})
        for seq, time, type_, data in rows
    ]


__all__ = ["SCHEMA_VERSION", "SqliteSink", "load_sqlite"]
