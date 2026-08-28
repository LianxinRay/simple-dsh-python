"""The Context mechanism in miniature: service registry + waterfall interception.

No agent, no LLM — just the two ideas that make the harness pluggable.
Run from the project root:

    python examples/context_demo.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_dsh.cordis import Context


async def main() -> None:
    ctx = Context()

    # --- 1. Service registry: look up by key, not by class import -----------
    class Greeter:
        def greet(self, name: str) -> str:
            return f"hello, {name}"

    ctx.service("greeter", Greeter())       # claim the ctx.greeter slot
    print(ctx.greeter.greet("world"))       # consumers know the key only

    # Swap the implementation — consumer code would not change.
    ctx.service("greeter", Greeter())
    ctx._services["greeter"] = type("Loud", (), {
        "greet": lambda self, name: f"HELLO, {name.upper()}!!!"
    })()
    print(ctx.greeter.greet("world"))

    # --- 2. Waterfall: intercept and transform a pipeline -------------------
    # Imagine "build/request" is a pipeline any plugin can hook.
    ctx.on("build/request", lambda req, next_: next_({**req, "retries": 3}))

    async def add_auth(req, next_):
        req = dict(req, headers={**req.get("headers", {}), "auth": "token-1"})
        return await next_(req)             # delegate; must call next_!

    async def short_circuit_if_dry_run(req, next_):
        if req.get("dry_run"):
            return {"status": "skipped"}    # no next_() -> chain stops here
        return await next_()

    ctx.on("build/request", add_auth)
    ctx.on("build/request", short_circuit_if_dry_run)

    result = await ctx.waterfall_terminal(
        "build/request",
        {"url": "/api/data"},
        terminal=lambda req: {"status": "sent", "request": req},
    )
    print(result)

    result = await ctx.waterfall_terminal(
        "build/request",
        {"url": "/api/data", "dry_run": True},
        terminal=lambda req: {"status": "sent", "request": req},
    )
    print(result)

    # --- 3. Effects: every registration unwinds ------------------------------
    seen = []
    dispose = ctx.on("build/request", lambda req, next_: seen.append(req) or next_())
    await ctx.waterfall("build/request", {"n": 1})
    dispose()                                # unregister exactly this listener
    await ctx.waterfall("build/request", {"n": 2})
    print(f"listener saw {len(seen)} request(s) before disposal")


if __name__ == "__main__":
    asyncio.run(main())
