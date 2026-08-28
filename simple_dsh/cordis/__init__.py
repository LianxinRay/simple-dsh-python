"""Cordis-style plugin runtime (P0-1).

A faithful, minimal Python port of the five Cordis ideas used by
DeepSeek Harness:

- A plugin is a function ``plugin(ctx, config)`` (with an optional
  ``inject`` attribute listing required service keys) or a ``Service``
  subclass whose lifecycle is mounted into a context.
- A context is a repository of services claimed under stable ``ctx.<key>``
  names; lookup walks up the parent chain.
- ``inject`` declares service dependencies; mounting waits until the
  required services exist and fails loud when they never will.
- Typed events dispatch as ``emit`` / ``waterfall`` / ``parallel`` /
  ``serial``.
- Registrations are reversible effects unwound on dispose.
"""

from .context import (
    Context,
    Disposer,
    InjectError,
    Listener,
    Service,
    maybe_await,
)

__all__ = [
    "Context",
    "Disposer",
    "InjectError",
    "Listener",
    "Service",
    "maybe_await",
]
