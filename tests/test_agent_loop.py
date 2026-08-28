"""Tests for the agent loop driver, end to end against a scripted adapter."""

import unittest

from simple_dsh.agent import REJECT, Agent
from simple_dsh.cordis import Context
from simple_dsh.llm import (
    LlmRegistry,
    MessageStop,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
)
from simple_dsh.prompts import PromptSection, SystemPrompt
from simple_dsh.tools import ToolDefinition, ToolRegistry, ToolResult


class ScriptedAdapter:
    """A fake LLM adapter that replays scripted chunk sequences in order."""

    provider = "scripted"
    models = ("scripted-model",)

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        for chunk in self._scripts.pop(0):
            yield chunk


def text_script(text, usage=TokenUsage(10, 5)):
    return [TextDelta(text), MessageStop(usage)]


def tool_call_script(call_id="c1", name="echo", arguments='{"text": "hi"}'):
    return [
        ToolCallStart(id=call_id, name=name),
        ToolCallDelta(id=call_id, arguments_delta=arguments),
        ToolCallEnd(call_id),
        MessageStop(TokenUsage(10, 5)),
    ]


async def make_ctx(adapter, with_tool=True) -> Context:
    ctx = Context()
    llm = LlmRegistry()
    llm.register_adapter(adapter)
    ctx.service("llm", llm)
    prompt = SystemPrompt()
    prompt.register_section(PromptSection(id="role", render=lambda: "You are helpful."))
    ctx.service("systemPrompt", prompt)
    tools = ToolRegistry(ctx)
    if with_tool:

        async def echo(args):
            return ToolResult.text("echo: " + args["text"])

        tools.register(
            ToolDefinition(
                name="echo",
                description="Echo text.",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                execute=echo,
            )
        )
    ctx.service("tools", tools)
    return ctx


def event_types(agent):
    return [e.type for e in agent.session]


class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    async def test_single_text_turn(self):
        adapter = ScriptedAdapter([text_script("Hello!")])
        agent = Agent(await make_ctx(adapter), model="scripted-model")
        await agent.prompt("hi")
        await agent.when_idle()

        self.assertEqual(
            event_types(agent),
            [
                "turn/start",
                "step/start",
                "user/message",
                "assistant/chunk",
                "assistant/chunk",
                "assistant/message",
                "step/end",
                "turn/end",
            ],
        )
        end = agent.session.at(len(agent.session) - 1)
        self.assertEqual(end.data["reason"], "completed")
        history = agent.session.derive_messages()
        self.assertEqual([m.role for m in history], ["user", "assistant"])
        self.assertEqual(history[1].content, (TextBlock("Hello!"),))

    async def test_tool_call_owes_a_second_step(self):
        adapter = ScriptedAdapter(
            [tool_call_script(), text_script("done", TokenUsage(20, 7))]
        )
        agent = Agent(await make_ctx(adapter), model="scripted-model")
        await agent.prompt("use the tool")
        await agent.when_idle()

        types = event_types(agent)
        self.assertEqual(types.count("step/start"), 2)
        self.assertEqual(types.count("tool/call"), 1)
        self.assertEqual(types.count("tool/result"), 1)
        self.assertEqual(types[-1], "turn/end")

        # The adapter saw the tool result in the second request's history.
        second_request = adapter.requests[1]
        last = second_request.messages[-1]
        self.assertIsInstance(last.content[0], ToolResultBlock)
        self.assertEqual(last.content[0].content, (TextBlock("echo: hi"),))
        # The first request carried the assembled system prompt.
        self.assertEqual(adapter.requests[0].system, "You are helpful.")
        # Usage travels with the assistant message event.
        usage_events = [
            e for e in agent.session if e.type == "assistant/message" and "usage" in e.data
        ]
        self.assertEqual(len(usage_events), 2)

    async def test_pre_step_rejection_closes_turn_with_no_step(self):
        adapter = ScriptedAdapter([text_script("never sent")])
        ctx = await make_ctx(adapter)

        async def reject(claimed, next_):
            return REJECT

        ctx.on("agent/pre-step", reject)
        agent = Agent(ctx, model="scripted-model")
        await agent.prompt("hi")
        await agent.when_idle()

        self.assertEqual(event_types(agent), ["turn/start", "turn/end"])
        self.assertEqual(agent.session.at(1).data["reason"], "rejected")
        self.assertEqual(adapter.requests, [])  # model was never called

    async def test_injected_context_waits_for_a_prompt(self):
        adapter = ScriptedAdapter([text_script("ok")])
        agent = Agent(await make_ctx(adapter), model="scripted-model")
        agent.inject("background fact")  # must not wake the driver
        await self._settle()
        self.assertEqual(event_types(agent), [])

        await agent.prompt("go")
        await agent.when_idle()
        user_events = [e for e in agent.session if e.type == "user/message"]
        self.assertEqual(len(user_events), 2)
        kinds = [e.data["source"]["kind"] for e in user_events]
        self.assertEqual(kinds, ["user", "plugin"])

    async def _settle(self):
        import asyncio

        await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
