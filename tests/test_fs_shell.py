"""Tests for the fs and shell tools."""

import sys
import tempfile
import unittest
from pathlib import Path

from simple_dsh.cordis import Context
from simple_dsh.llm import ToolCallBlock
from simple_dsh.tools import ToolRegistry
from simple_dsh.tools.fs_tools import WorkspacePolicy, register_fs_tools
from simple_dsh.tools.shell import register_shell_tool


def call(name, **args):
    import json

    return ToolCallBlock(id="c1", name=name, arguments=json.dumps(args))


class TestFsTools(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = Context()
        self.tools = ToolRegistry(self.ctx)
        self.ctx.service("tools", self.tools)
        register_fs_tools(self.tools, self.root)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_write_read_edit_list_cycle(self):
        r = await self.tools.execute(call("write_file", path="notes/a.txt", content="one\ntwo\nthree"))
        self.assertFalse(r.is_error)

        r = await self.tools.execute(call("read_file", path="notes/a.txt"))
        self.assertIn("1\tone", r.content[0].text)
        r = await self.tools.execute(call("read_file", path="notes/a.txt", offset=2, limit=1))
        self.assertEqual(r.content[0].text, "2\ttwo")

        r = await self.tools.execute(
            call("edit_file", path="notes/a.txt", old_string="two", new_string="TWO")
        )
        self.assertFalse(r.is_error)
        self.assertIn("TWO", (self.root / "notes/a.txt").read_text())

        r = await self.tools.execute(call("list_directory", path="notes"))
        self.assertIn("a.txt", r.content[0].text)

    async def test_edit_requires_unique_match(self):
        (self.root / "d.txt").write_text("dup dup", encoding="utf-8")
        r = await self.tools.execute(
            call("edit_file", path="d.txt", old_string="dup", new_string="x")
        )
        self.assertTrue(r.is_error)
        self.assertIn("not unique", r.content[0].text)

    async def test_path_escape_is_refused(self):
        policy = WorkspacePolicy(self.root)
        with self.assertRaises(ValueError):
            policy.resolve("../outside.txt")
        # Through the pipeline, the escape normalizes to is_error.
        r = await self.tools.execute(call("read_file", path="../outside.txt"))
        self.assertTrue(r.is_error)
        self.assertIn("escapes", r.content[0].text)


class TestShellTool(unittest.IsolatedAsyncioTestCase):
    async def test_runs_command_and_captures_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context()
            tools = ToolRegistry(ctx)
            ctx.service("tools", tools)
            register_shell_tool(tools, tmp)
            r = await tools.execute(
                call("bash", command=f'"{sys.executable}" -c "print(\'hello-shell\')"')
            )
            self.assertFalse(r.is_error)
            self.assertIn("hello-shell", r.content[0].text)
            self.assertEqual(r.meta["exit_code"], 0)

    async def test_nonzero_exit_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context()
            tools = ToolRegistry(ctx)
            ctx.service("tools", tools)
            register_shell_tool(tools, tmp)
            r = await tools.execute(
                call("bash", command=f'"{sys.executable}" -c "import sys; sys.exit(3)"')
            )
            self.assertTrue(r.is_error)
            self.assertEqual(r.meta["exit_code"], 3)


if __name__ == "__main__":
    unittest.main()
