"""End-to-end demo: a full turn with reasoning, a tool call, and persistence.

Run from the project root:

    python examples/echo_demo.py

A scripted adapter stands in for a real LLM provider: it first streams
reasoning + text + a ``word_count`` tool call, then — once the tool result
is logged — streams the final answer. The whole turn is printed from the
session log, which is also written to ``examples/out/session.jsonl``.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_dsh.agent import Agent
from simple_dsh.cordis import Context
from simple_dsh.llm import (
    LlmRegistry,
    MessageStop,
    ReasoningDelta,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from simple_dsh.prompts import PromptSection, SystemPrompt
from simple_dsh.session import JsonlSink, load_jsonl, Session
from simple_dsh.tools import ToolDefinition, ToolRegistry, ToolResult


class ScriptedAdapter:
    """Stands in for a provider adapter: replays scripted chunk streams."""

    provider = "scripted"
    models = ("scripted-model",)

    def __init__(self, scripts):
        self._scripts = list(scripts)

    async def stream(self, request):
        print(f"  [adapter] request: {len(request.messages)} messages, "
              f"{len(request.tools)} tool schema(s)")
        for chunk in self._scripts.pop(0):
            yield chunk


async def main() -> None:
    ctx = Context()

    llm = LlmRegistry()
    llm.register_adapter(ScriptedAdapter([
        # Step 1: reason, speak, call the tool.
        [
            ReasoningDelta("The user wants a word count. Use the tool."),
            TextDelta("Let me count those words. "),
            ToolCallStart(id="c1", name="word_count"),
            ToolCallDelta(id="c1", arguments_delta='{"text": "the quick brown fox"}'),
            ToolCallEnd("c1"),
            MessageStop(TokenUsage(42, 18)),
        ],
        # Step 2 (continuation): final answer after the tool result.
        [TextDelta("The sentence has 4 words."), MessageStop(TokenUsage(55, 9))],
    ]))
    ctx.service("llm", llm)

    prompt = SystemPrompt()
    prompt.register_section(
        PromptSection(id="role", priority=10, render=lambda: "You are a careful assistant.")
    )
    ctx.service("systemPrompt", prompt)

    tools = ToolRegistry(ctx)

    async def word_count(args):
        return ToolResult.text(str(len(args["text"].split())))

    tools.register(ToolDefinition(
        name="word_count",
        description="Count the words in a text.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=word_count,
    ))
    ctx.service("tools", tools)

    out = Path(__file__).parent / "out" / "session.jsonl"
    out.unlink(missing_ok=True)
    sink = JsonlSink(out)
    session = Session()
    session.add_sink(sink)

    agent = Agent(ctx, session=session, model="scripted-model")
    agent.inject("runtime note: demo workspace")  # waits for the prompt
    await agent.prompt("How many words in 'the quick brown fox'?")
    await agent.when_idle()
    sink.close()

    print("\n=== session log ===")
    for event in session:
        print(f"  seq={event.seq:<2} {event.type}")

    print("\n=== derived model history ===")
    for message in session.derive_messages():
        blocks = ", ".join(
            getattr(b, "text", None) or getattr(b, "thinking", None)
            or getattr(b, "name", None) or b.type
            for b in message.content
        )
        print(f"  {message.role:<9} [{message.source.kind}] {blocks}")

    replayed = Session(load_jsonl(out))
    assert [e.type for e in replayed] == [e.type for e in session]
    print(f"\nReplayed {len(replayed)} events from {out} — identical log.")


if __name__ == "__main__":
    asyncio.run(main())
