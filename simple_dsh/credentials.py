"""Credential resolution (P1).

Ported from ``packages/credentials``: env-over-``.env`` — real environment
variables win, the ``.env`` file fills the gaps. Misconfiguration fails
loud: :meth:`Credentials.require` raises rather than returning ``None``.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a ``.env`` file: ``KEY=VALUE`` lines, ``#`` comments, quotes."""
    values: dict[str, str] = {}
    text = Path(path).read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


class Credentials:
    """Resolves credential references: environment first, then ``.env``."""

    def __init__(self, env_path: str | Path | None = None) -> None:
        self._file_values: dict[str, str] = {}
        if env_path is not None and Path(env_path).is_file():
            self._file_values = parse_env_file(env_path)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Return the credential for ``key``; ``os.environ`` wins."""
        return os.environ.get(key, self._file_values.get(key, default))

    def require(self, key: str) -> str:
        """Return the credential for ``key`` or raise — never silently miss."""
        value = self.get(key)
        if not value:
            raise KeyError(f"credential {key!r} not found in environment or .env")
        return value
