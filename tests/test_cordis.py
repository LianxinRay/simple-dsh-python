"""Tests for the cordis plugin runtime."""

import unittest

from simple_dsh.cordis import Context, InjectError, Service


class TestServices(unittest.IsolatedAsyncioTestCase):
    async def test_claim_and_lookup_walks_up(self):
        root = Context()
        dispose = root.service("tools", {"name": "tools"})
        child = root.scope("child")
        self.assertEqual(child.tools["name"], "tools")
        dispose()
        with self.assertRaises(AttributeError):
            child.tools

    async def test_missing_service_raises_keyerror(self):
        ctx = Context()
        with self.assertRaises(KeyError):
            ctx.service("nope")

    async def test_inject_controls_mount_order(self):
        order = []

        class DbService(Service):
            name = "db"

            async def start(self, ctx):
                order.append("db")

        def app_plugin(ctx, config):
            order.append("app")

        app_plugin.inject = ("db",)

        root = Context()
        root.use(app_plugin)  # declared first, mounts second
        root.use(DbService)
        await root.ready()
        self.assertEqual(order, ["db", "app"])
        self.assertIsInstance(root.service("db"), DbService)

    async def test_unsatisfiable_inject_fails_loud(self):
        def needy(ctx, config):
            pass

        needy.inject = ("never-registered",)
        root = Context()
        root.use(needy)
        with self.assertRaises(InjectError):
            await root.ready()

    async def test_service_disposed_with_plugin_scope(self):
        class Cache(Service):
            name = "cache"

        root = Context()
        root.use(Cache)
        await root.ready()
        self.assertIsNotNone(root._lookup_service("cache"))
        await root.dispose()
        self.assertIsNone(root._lookup_service("cache"))


class TestEvents(unittest.IsolatedAsyncioTestCase):
    async def test_emit_observes_in_order(self):
        ctx = Context()
        seen = []
        ctx.on("e", lambda v: seen.append(("a", v)))
        ctx.on("e", lambda v: seen.append(("b", v)))
        await ctx.emit("e", 1)
        self.assertEqual(seen, [("a", 1), ("b", 1)])

    async def test_child_listeners_run_before_parent(self):
        root = Context()
        child = root.scope("c")
        seen = []
        root.on("e", lambda: seen.append("root"))
        child.on("e", lambda: seen.append("child"))
        await child.emit("e")
        self.assertEqual(seen, ["child", "root"])

    async def test_parallel_collects_nothing_and_waits(self):
        ctx = Context()
        seen = []
        ctx.on("e", lambda v: seen.append(v + 1))
        ctx.on("e", lambda v: seen.append(v + 2))
        await ctx.parallel("e", 0)
        self.assertEqual(sorted(seen), [1, 2])

    async def test_serial_collects_results_in_order(self):
        ctx = Context()
        ctx.on("e", lambda v: v * 2)
        ctx.on("e", lambda v: v * 3)
        self.assertEqual(await ctx.serial("e", 2), [4, 6])

    async def test_waterfall_delegates_through_next(self):
        ctx = Context()

        async def double(value, next_):
            return (await next_()) * 2

        async def plus_one(value, next_):
            return await next_(value + 1)

        ctx.on("e", double)
        ctx.on("e", plus_one)
        # double(10) -> next() -> plus_one(10) -> next(11) -> terminal(11) = 11
        # plus_one returns 11 -> double returns 22
        self.assertEqual(await ctx.waterfall("e", 10), 22)

    async def test_waterfall_short_circuit_without_next(self):
        ctx = Context()
        called = []

        async def policy(value, next_):
            return "denied"  # no next(): short-circuits

        async def never(value, next_):
            called.append(True)
            return await next_()

        ctx.on("e", policy)
        ctx.on("e", never)
        self.assertEqual(await ctx.waterfall("e", "request"), "denied")
        self.assertEqual(called, [])

    async def test_waterfall_mutation_without_next_keeps_payload(self):
        ctx = Context()

        async def annotate(value, next_):
            value["annotated"] = True  # returns None, no next(): keep mutation
            return None

        ctx.on("e", annotate)
        self.assertEqual(await ctx.waterfall("e", {}), {"annotated": True})

    async def test_waterfall_terminal_runs_at_innermost(self):
        ctx = Context()

        async def around(value, next_):
            return f"[{await next_()}]"

        ctx.on("e", around)
        result = await ctx.waterfall_terminal("e", "x", lambda v: v.upper())
        self.assertEqual(result, "[X]")

    async def test_listener_disposer_removes(self):
        ctx = Context()
        seen = []
        dispose = ctx.on("e", lambda: seen.append(1))
        dispose()
        await ctx.emit("e")
        self.assertEqual(seen, [])


class TestEffects(unittest.IsolatedAsyncioTestCase):
    async def test_effects_unwind_in_reverse_order(self):
        ctx = Context()
        log = []
        ctx.effect(lambda: lambda: log.append("first"))
        ctx.effect(lambda: lambda: log.append("second"))
        await ctx.dispose()
        self.assertEqual(log, ["second", "first"])

    async def test_children_dispose_before_parent(self):
        root = Context()
        child = root.scope("c")
        log = []
        root.effect(lambda: lambda: log.append("root"))
        child.effect(lambda: lambda: log.append("child"))
        await root.dispose()
        self.assertEqual(log, ["child", "root"])


if __name__ == "__main__":
    unittest.main()
