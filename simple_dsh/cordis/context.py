"""Context, service registry, typed events, and effect lifecycle."""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable, ClassVar

Disposer = Callable[[], Any]
"""Undoes one registration; may be sync or async."""

Listener = Callable[..., Awaitable[Any]]
"""Event listener. Waterfall listeners receive ``(payload, next)``."""

_SENTINEL = object()


async def maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it."""
    if inspect.isawaitable(value):
        return await value
    return value


class InjectError(RuntimeError):
    """Raised when pending plugins can never see their required services."""


class Service:
    """Base class for plugins that claim a stable ``ctx.<name>`` key.

    Subclasses set :attr:`name` and optionally :attr:`inject`; ``start`` and
    ``stop`` are the lifecycle hooks. The instance becomes visible as
    ``ctx.<name>`` on the context that mounted it once ``start`` finishes.
    """

    name: ClassVar[str]
    inject: ClassVar[tuple[str, ...]] = ()

    async def start(self, ctx: Context) -> None:  # noqa: D102
        return None

    async def stop(self) -> None:  # noqa: D102
        return None


class Context:
    """A repository of services, event listeners, and reversible effects.

    Contexts form a tree: service lookup and event listener collection walk
    from a child up to the root, so agent-scoped registrations shadow and
    precede global ones. Disposing a context disposes its children first,
    then its own effects in reverse registration order.
    """

    def __init__(self, parent: Context | None = None, name: str = "root") -> None:
        self.parent = parent
        self.name = name
        self._services: dict[str, Any] = {}
        self._listeners: dict[str, list[Listener]] = defaultdict(list)
        self._disposers: list[Disposer] = []
        self._children: list[Context] = []
        self._pending: list[tuple[Callable[..., Any], Any]] = []
        self._disposed = False

    # ------------------------------------------------------------------ tree

    def scope(self, name: str) -> Context:
        """Create a child context for scoped (e.g. per-plugin) registrations."""
        child = Context(parent=self, name=name)
        self._children.append(child)
        return child

    # -------------------------------------------------------------- services

    def _lookup_service(self, key: str) -> Any:
        ctx: Context | None = self
        while ctx is not None:
            if key in ctx._services:
                return ctx._services[key]
            ctx = ctx.parent
        return None

    def service(self, key: str, instance: Any = _SENTINEL) -> Any:
        """Get a service, or claim ``ctx.<key>`` with ``instance``.

        Claiming returns a disposer that withdraws the key. Getting a missing
        key raises :class:`KeyError` — misconfiguration fails loud.
        """
        if instance is _SENTINEL:
            svc = self._lookup_service(key)
            if svc is None:
                raise KeyError(f"no service registered under ctx.{key!r}")
            return svc
        self._services[key] = instance
        return self._track(lambda: self._services.pop(key, None))

    def __getattr__(self, key: str) -> Any:
        # Only reached when normal attribute lookup fails.
        if key.startswith("_"):
            raise AttributeError(key)
        svc = self._lookup_service(key)
        if svc is None:
            raise AttributeError(f"no service registered under ctx.{key!r}")
        return svc

    # ---------------------------------------------------------------- events

    def on(self, event: str, listener: Listener, *, prepend: bool = False) -> Disposer:
        """Listen to ``event``. Returns the disposer removing the listener."""
        listeners = self._listeners[event]
        if prepend:
            listeners.insert(0, listener)
        else:
            listeners.append(listener)

        def dispose() -> None:
            if listener in listeners:
                listeners.remove(listener)

        return self._track(dispose)

    def _chain_listeners(self, event: str) -> list[Listener]:
        """Listeners from this context up to the root, child first."""
        chain: list[Listener] = []
        ctx: Context | None = self
        while ctx is not None:
            chain.extend(ctx._listeners.get(event, ()))
            ctx = ctx.parent
        return chain

    async def emit(self, event: str, *args: Any) -> None:
        """Notify listeners in registration order; return values are ignored."""
        for listener in self._chain_listeners(event):
            await maybe_await(listener(*args))

    async def parallel(self, event: str, *args: Any) -> None:
        """Run all listeners concurrently and wait for every one."""
        await asyncio.gather(
            *(maybe_await(listener(*args)) for listener in self._chain_listeners(event))
        )

    async def serial(self, event: str, *args: Any) -> list[Any]:
        """Run listeners in registration order, collecting return values."""
        results = []
        for listener in self._chain_listeners(event):
            results.append(await maybe_await(listener(*args)))
        return results

    async def waterfall(self, event: str, payload: Any) -> Any:
        """Around-middleware: each listener receives ``(payload, next)``.

        A listener MUST call ``next()`` to delegate; returning without it
        short-circuits the chain. A non-``None`` return value replaces the
        payload for upstream listeners; returning ``None`` after delegating
        propagates the delegated result, and returning ``None`` without
        delegating keeps the (possibly mutated) payload.
        """
        return await self.waterfall_terminal(event, payload, lambda value: value)

    async def waterfall_terminal(
        self,
        event: str,
        payload: Any,
        terminal: Callable[[Any], Any],
    ) -> Any:
        """Waterfall whose innermost step calls ``terminal(payload)``.

        Used by the tool pipeline, where the tool body sits at the end of the
        ``tools/execute`` around-chain.
        """
        listeners = self._chain_listeners(event)

        async def call_at(index: int, value: Any) -> Any:
            if index == len(listeners):
                return await maybe_await(terminal(value))
            delegated = False
            delegated_result: Any = _SENTINEL

            async def next_(new_value: Any = _SENTINEL) -> Any:
                nonlocal delegated, delegated_result
                delegated = True
                delegated_result = await call_at(
                    index + 1, value if new_value is _SENTINEL else new_value
                )
                return delegated_result

            result = await maybe_await(listeners[index](value, next_))
            if result is not None:
                return result
            if delegated:
                return delegated_result
            return value

        return await call_at(0, payload)

    # --------------------------------------------------------------- effects

    def _track(self, disposer: Disposer) -> Disposer:
        self._disposers.append(disposer)

        def dispose_once() -> Any:
            if disposer in self._disposers:
                self._disposers.remove(disposer)
                return disposer()
            return None

        return dispose_once

    def effect(self, fn: Callable[[], Disposer | None]) -> Disposer:
        """Register an effect; its returned callable undoes it on dispose."""
        disposer = fn()
        return self._track(disposer or (lambda: None))

    # --------------------------------------------------------------- plugins

    def use(self, plugin: Any, config: Any = None) -> None:
        """Queue a plugin for mounting once its ``inject`` services exist."""
        self._pending.append((plugin, config))

    async def ready(self) -> None:
        """Mount pending plugins in dependency order; fail loud if stuck."""
        while self._pending:
            progressed = False
            for entry in list(self._pending):
                plugin, _ = entry
                inject = getattr(plugin, "inject", ())
                if all(self._lookup_service(key) is not None for key in inject):
                    self._pending.remove(entry)
                    await self._mount(*entry)
                    progressed = True
            if not progressed:
                missing = {
                    key
                    for plugin, _ in self._pending
                    for key in getattr(plugin, "inject", ())
                    if self._lookup_service(key) is None
                }
                raise InjectError(
                    f"plugins cannot mount, missing services: {sorted(missing)}"
                )

    async def _mount(self, plugin: Any, config: Any) -> None:
        name = getattr(plugin, "name", None) or getattr(plugin, "__name__", "plugin")
        child = self.scope(str(name))
        if inspect.isclass(plugin) and issubclass(plugin, Service):
            instance = plugin()
            await maybe_await(instance.start(child))
            # The service key lives on the mounting context so siblings can
            # resolve it; disposal is tracked on the plugin's own scope.
            key = instance.name
            self._services[key] = instance

            def withdraw(instance: Any = instance, key: str = key) -> Any:
                self._services.pop(key, None)
                return instance.stop()

            child._track(withdraw)
        else:
            await maybe_await(plugin(child, config))

    # --------------------------------------------------------------- teardown

    async def dispose(self) -> None:
        """Dispose children first, then own effects in reverse order."""
        if self._disposed:
            return
        self._disposed = True
        for child in reversed(self._children):
            await child.dispose()
        while self._disposers:
            disposer = self._disposers.pop()
            await maybe_await(disposer())
