"""Tests for system-prompt assembly and the tool pipeline."""

import unittest

from simple_dsh.cordis import Context
from simple_dsh.llm import TextBlock, ToolCallBlock
from simple_dsh.prompts import PromptSection, SystemPrompt
from simple_dsh.tools import ToolDefinition, ToolRegistry, ToolResult


class TestSystemPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_sections_render_in_priority_order(self):
        sp = SystemPrompt()
        sp.register_section(PromptSection(id="late", render=lambda: "L", priority=200))
        sp.register_section(PromptSection(id="early", render=lambda: "E", priority=10))
        self.assertEqual(await sp.assemble(), "E\n\nL")

    async def test_empty_sections_are_skipped_and_disposal_removes(self):
        sp = SystemPrompt()
        sp.register_section(PromptSection(id="empty", render=lambda: ""))
        dispose = sp.register_section(PromptSection(id="kept", render=lambda: "K"))
        self.assertEqual(await sp.assemble(), "K")
        dispose()
        self.assertEqual(await sp.assemble(), "")


def echo_tool() -> ToolDefinition:
    async def body(args):
        return ToolResult.text(args["text"].upper())

    return ToolDefinition(
        name="echo",
        description="Echo text, uppercased.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        execute=body,
    )


def make_call(name="echo", arguments='{"text": "hi"}') -> ToolCallBlock:
    return ToolCallBlock(id="c1", name=name, arguments=arguments)


class TestToolPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ctx = Context()
        self.tools = ToolRegistry(self.ctx)
        self.ctx.service("tools", self.tools)
        self.tools.register(echo_tool())

    async def test_happy_path_parses_args_and_runs_body(self):
        result = await self.tools.execute(make_call())
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, [TextBlock("HI")])

    async def test_schemas_join_prompt_assembly(self):
        (schema,) = self.tools.schemas()
        self.assertEqual(schema["name"], "echo")
        self.assertIn("text", schema["parameters"]["properties"])

    async def test_unknown_tool_is_error_not_exception(self):
        result = await self.tools.execute(make_call(name="ghost"))
        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", result.content[0].text)

    async def test_invalid_arguments_json_is_error(self):
        result = await self.tools.execute(make_call(arguments="{oops"))
        self.assertTrue(result.is_error)

    async def test_pre_execute_waterfall_can_deny(self):
        async def deny(state, next_):
            state.action = "deny"
            state.reason = "off limits"
            return state  # short-circuit with the mutated decision

        self.ctx.on("tools/pre-execute", deny)
        result = await self.tools.execute(make_call())
        self.assertTrue(result.is_error)
        self.assertIn("off limits", result.content[0].text)

    async def test_guard_denies_and_cannot_be_reallowed(self):
        self.tools.register_guard(lambda call: "deny", name="test-guard")
        result = await self.tools.execute(make_call())
        self.assertTrue(result.is_error)
        self.assertIn("test-guard", result.content[0].text)

    async def test_execute_waterfall_wraps_the_body(self):
        seen = []

        async def around(call, next_):
            seen.append(call.name)
            return await next_()

        self.ctx.on("tools/execute", around)
        result = await self.tools.execute(make_call())
        self.assertEqual(seen, ["echo"])
        self.assertEqual(result.content, [TextBlock("HI")])

    async def test_post_execute_can_replace_result(self):
        async def replace(result, next_):
            await next_()
            return ToolResult.text("replaced")

        self.ctx.on("tools/post-execute", replace)
        result = await self.tools.execute(make_call())
        self.assertEqual(result.content, [TextBlock("replaced")])

    async def test_body_exception_normalizes_to_is_error(self):
        async def boom(args):
            raise RuntimeError("kaboom")

        self.tools.register(
            ToolDefinition(name="boom", description="fails", parameters={}, execute=boom)
        )
        result = await self.tools.execute(make_call(name="boom", arguments="{}"))
        self.assertTrue(result.is_error)
        self.assertIn("kaboom", result.content[0].text)


if __name__ == "__main__":
    unittest.main()
