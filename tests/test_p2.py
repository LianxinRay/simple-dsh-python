"""Tests for P2: approval, compaction, todo, subagent, web, sqlite, preset."""

import json
import tempfile
import unittest
from pathlib import Path

from simple_dsh.agent import Agent
from simple_dsh.approval import ApprovalService, require_approval
from simple_dsh.compaction import CompactionService, estimate_tokens
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
)
from simple_dsh.llm.types import ToolCallBlock
from simple_dsh.preset import DEFAULT_PRESET, merge_preset
from simple_dsh.prompts import SystemPrompt
from simple_dsh.session import Session, SqliteSink, load_sqlite, make_user_message
from simple_dsh.tools import ToolDefinition, ToolRegistry, ToolResult
from simple_dsh.tools.subagent import register_subagent_tool
from simple_dsh.tools.todo import latest_todos, register_todo_tool
from simple_dsh.tools.web import html_to_text


def make_ctx() -> Context:
    ctx = Context()
    ctx.service("systemPrompt", SystemPrompt())
    tools = ToolRegistry(ctx)
    ctx.service("tools", tools)
    return ctx


def write_call(name="write_file", arguments='{"path": "a.txt"}'):
    return ToolCallBlock(id="c1", name=name, arguments=arguments)


class TestApproval(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ctx = make_ctx()
        self.tools = self.ctx.service("tools")

        async def body(args):
            return ToolResult.text("wrote")

        self.tools.register(
            ToolDefinition(name="write_file", description="w", parameters={}, execute=body)
        )

    async def test_allowed_when_responder_says_yes(self):
        approval = ApprovalService(responder=lambda call: True)
        require_approval(self.ctx, approval, ["write_file"])
        result = await self.tools.execute(write_call())
        self.assertFalse(result.is_error)

    async def test_denied_when_responder_says_no(self):
        approval = ApprovalService(responder=lambda call: False)
        require_approval(self.ctx, approval, ["write_file"])
        result = await self.tools.execute(write_call())
        self.assertTrue(result.is_error)
        self.assertIn("approval", result.content[0].text)

    async def test_fail_closed_without_responder(self):
        approval = ApprovalService()  # no responder: unanswerable -> deny
        require_approval(self.ctx, approval, ["write_file"])
        result = await self.tools.execute(write_call())
        self.assertTrue(result.is_error)

    async def test_ungated_tool_bypasses_approval(self):
        approval = ApprovalService()
        require_approval(self.ctx, approval, ["bash"])  # gates bash, not write_file
        result = await self.tools.execute(write_call())
        self.assertFalse(result.is_error)


class ScriptedAdapter:
    provider = "scripted"
    models = ("m",)

    def __init__(self, scripts):
        self._scripts = list(scripts)

    async def stream(self, request):
        for chunk in self._scripts.pop(0):
            yield chunk


class TestCompaction(unittest.IsolatedAsyncioTestCase):
    async def test_over_budget_folds_old_history_into_summary(self):
        session = Session()
        for i in range(8):
            session.append("user/message", _msg(f"old message {i} " + "x" * 100))
        session.append("user/message", _msg("recent"))

        async def summarizer(messages):
            return f"summary of {len(messages)} messages"

        service = CompactionService(summarizer, max_tokens=50, keep_recent=2)
        self.assertTrue(await service.maybe_compact(session))

        derived = session.derive_messages()
        self.assertTrue(derived[0].content[0].text.startswith("[conversation summary]"))
        self.assertEqual(len(derived), 3)  # summary + 2 kept
        types = [e.type for e in session]
        self.assertEqual(types[-3:], ["compaction/start", "compaction/summary", "compaction/end"])

    async def test_under_budget_is_noop(self):
        session = Session()
        session.append("user/message", _msg("short"))
        service = CompactionService(None, max_tokens=10_000)
        self.assertFalse(await service.maybe_compact(session))

    def test_estimate_tokens_counts_text(self):
        messages = [message for message in
                    [make_user_message("a" * 400)]]
        self.assertEqual(estimate_tokens(messages), 100)


def _msg(text):
    import dataclasses

    from simple_dsh.llm import message_to_json
    return message_to_json(make_user_message(text))


class TestTodo(unittest.IsolatedAsyncioTestCase):
    async def test_write_logs_snapshot_and_replays_latest(self):
        ctx = make_ctx()
        session = Session()
        register_todo_tool(ctx.service("tools"), session)
        call = ToolCallBlock(
            id="c1", name="todo_write",
            arguments=json.dumps({"todos": [
                {"title": "task A", "status": "done"},
                {"title": "task B", "status": "pending"},
            ]}),
        )
        result = await ctx.service("tools").execute(call)
        self.assertFalse(result.is_error)
        events = [e for e in session if e.type == "todo/write"]
        self.assertEqual(len(events), 1)
        self.assertEqual(latest_todos(session)[1]["title"], "task B")
        # Log-only: never enters derived history.
        self.assertEqual(session.derive_messages(), [])

    async def test_invalid_status_rejected(self):
        ctx = make_ctx()
        register_todo_tool(ctx.service("tools"), Session())
        call = ToolCallBlock(
            id="c1", name="todo_write",
            arguments=json.dumps({"todos": [{"title": "x", "status": "weird"}]}),
        )
        self.assertTrue((await ctx.service("tools").execute(call)).is_error)


class TestSubagent(unittest.IsolatedAsyncioTestCase):
    async def test_delegate_runs_child_agent_and_returns_final_text(self):
        ctx = make_ctx()
        llm = LlmRegistry()
        llm.register_adapter(ScriptedAdapter([
            [TextDelta("parent thinking"), ToolCallStart(id="d1", name="delegate"),
             ToolCallDelta(id="d1", arguments_delta='{"task": "say hi"}'),
             ToolCallEnd("d1"), MessageStop(TokenUsage(1, 1))],
            [TextDelta("hi from child"), MessageStop(TokenUsage(1, 1))],
            [TextDelta("child said hi"), MessageStop(TokenUsage(1, 1))],
        ]))
        ctx.service("llm", llm)
        register_subagent_tool(ctx.service("tools"), ctx, model="m")

        agent = Agent(ctx, model="m")
        await agent.prompt("delegate something")
        await agent.when_idle()

        results = [e for e in agent.session if e.type == "tool/result"]
        self.assertEqual(len(results), 1)
        from simple_dsh.llm import message_from_json
        msg = message_from_json(results[0].data["message"])
        self.assertEqual(msg.content[0].content[0].text, "hi from child")
        self.assertEqual(results[0].data["meta"]["subagent_steps"], 1)


class TestWeb(unittest.TestCase):
    def test_html_to_text_strips_scripts(self):
        html = "<html><head><style>x{}</style></head><body>" \
               "<h1>Title</h1><script>var evil=1;</script><p>Body text</p></body></html>"
        text = html_to_text(html)
        self.assertIn("Title", text)
        self.assertIn("Body text", text)
        self.assertNotIn("evil", text)


class TestSqlite(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            session = Session()
            sink = SqliteSink(path)
            session.add_sink(sink)
            session.append("user/message", _msg("hello sqlite"))
            session.append("turn/end", {"turn": 1, "reason": "completed"})
            sink.close()

            events = load_sqlite(path)
            self.assertEqual([e.type for e in events], ["user/message", "turn/end"])
            restored = Session(events)
            self.assertEqual(
                restored.derive_messages()[0].content[0], TextBlock("hello sqlite")
            )

    def test_schema_version_mismatch_rejected(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            SqliteSink(path).close()
            db = sqlite3.connect(str(path))
            db.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
            db.commit()
            db.close()
            with self.assertRaises(ValueError):
                SqliteSink(path)


class TestPreset(unittest.TestCase):
    def test_merge_over_defaults(self):
        spec = merge_preset({"model": "deepseek-reasoner", "tools": {"shell": False}})
        self.assertEqual(spec["model"], "deepseek-reasoner")
        self.assertFalse(spec["tools"]["shell"])
        self.assertTrue(spec["tools"]["fs"])  # untouched defaults survive

    def test_default_preset_is_self_contained(self):
        self.assertIn("model", DEFAULT_PRESET)
        self.assertIn("timeout", DEFAULT_PRESET["guards"])


if __name__ == "__main__":
    unittest.main()
