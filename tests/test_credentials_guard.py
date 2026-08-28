"""Tests for credentials and the guard layer."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from simple_dsh.cordis import Context
from simple_dsh.credentials import Credentials, parse_env_file
from simple_dsh.guard import RepeatCallGuard, register_timeout
from simple_dsh.llm import ToolCallBlock
from simple_dsh.tools import ToolDefinition, ToolRegistry, ToolResult


class TestCredentials(unittest.TestCase):
    def test_parse_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\n"
                "A=1\n"
                'B="quoted"\n'
                "C='single'\n"
                "export D=4\n"
                "BROKEN LINE\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_env_file(path), {"A": "1", "B": "quoted", "C": "single", "D": "4"}
            )

    def test_environment_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("TOKEN=from-file\n", encoding="utf-8")
            creds = Credentials(path)
            self.assertEqual(creds.get("TOKEN"), "from-file")
            os.environ["TOKEN"] = "from-env"
            try:
                self.assertEqual(creds.get("TOKEN"), "from-env")
            finally:
                del os.environ["TOKEN"]

    def test_require_fails_loud(self):
        creds = Credentials()
        with self.assertRaises(KeyError):
            creds.require("DEFINITELY_MISSING_CREDENTIAL")


def slow_tool() -> ToolDefinition:
    async def body(args):
        await asyncio.sleep(5)
        return ToolResult.text("done")

    return ToolDefinition(name="slow", description="sleeps", parameters={}, execute=body)


class TestGuards(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ctx = Context()
        self.tools = ToolRegistry(self.ctx)
        self.ctx.service("tools", self.tools)

    async def test_timeout_normalizes_to_is_error(self):
        self.tools.register(slow_tool())
        register_timeout(self.ctx, 0.05)
        call = ToolCallBlock(id="c1", name="slow", arguments="{}")
        result = await self.tools.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("TimeoutError", result.content[0].text)

    async def test_fast_tool_unaffected_by_timeout(self):
        async def body(args):
            return ToolResult.text("quick")

        self.tools.register(
            ToolDefinition(name="quick", description="fast", parameters={}, execute=body)
        )
        register_timeout(self.ctx, 5)
        result = await self.tools.execute(ToolCallBlock(id="c1", name="quick", arguments="{}"))
        self.assertFalse(result.is_error)

    async def test_repeat_call_guard_denies_past_limit(self):
        guard = RepeatCallGuard(limit=3)
        guard.register(self.tools)

        async def body(args):
            return ToolResult.text("ok")

        self.tools.register(
            ToolDefinition(name="spin", description="repeats", parameters={}, execute=body)
        )
        call = ToolCallBlock(id="c", name="spin", arguments='{"x": 1}')
        results = [await self.tools.execute(call) for _ in range(4)]
        self.assertFalse(results[0].is_error)
        self.assertFalse(results[2].is_error)
        self.assertTrue(results[3].is_error)
        self.assertIn("repeat-call", results[3].content[0].text)
        # A different call resets the streak.
        other = ToolCallBlock(id="c2", name="spin", arguments='{"x": 2}')
        self.assertFalse((await self.tools.execute(other)).is_error)


if __name__ == "__main__":
    unittest.main()
