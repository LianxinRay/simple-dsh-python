"""Tests for the DeepSeek adapter: wire mapping, SSE translation, streaming."""

import unittest

from simple_dsh.llm import (
    MessageStop,
    ModelRequest,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    message_from_json,
    message_to_json,
)
from simple_dsh.llm.deepseek import (
    AdapterError,
    DeepSeekAdapter,
    OpenAiStreamTranslator,
    read_sse_payloads,
    request_to_openai,
)
from simple_dsh.session import make_tool_result_message, make_user_message


def assistant_with_tool_call():
    from simple_dsh.llm import AssistantProvenance, Message, MessageSource

    return Message(
        role="assistant",
        content=(ToolCallBlock(id="c1", name="echo", arguments='{"text":"hi"}'),),
        source=MessageSource(kind="model", provenance=AssistantProvenance("deepseek", "m")),
    )


class TestRequestMapping(unittest.TestCase):
    def test_roles_tools_and_system_map_to_openai(self):
        request = ModelRequest(
            model="deepseek-chat",
            system="Be helpful.",
            messages=(
                make_user_message("hello"),
                assistant_with_tool_call(),
                make_tool_result_message("c1", [TextBlock("HI")], False),
            ),
            tools=({"name": "echo", "description": "d", "parameters": {}},),
        )
        body = request_to_openai(request)
        self.assertEqual(body["model"], "deepseek-chat")
        self.assertTrue(body["stream"])
        self.assertEqual(body["messages"][0], {"role": "system", "content": "Be helpful."})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "hello"})
        assistant = body["messages"][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertIsNone(assistant["content"])
        self.assertEqual(
            assistant["tool_calls"][0]["function"],
            {"name": "echo", "arguments": '{"text":"hi"}'},
        )
        self.assertEqual(
            body["messages"][3],
            {"role": "tool", "tool_call_id": "c1", "content": "HI"},
        )
        self.assertEqual(body["tools"][0]["function"]["name"], "echo")


class TestTranslator(unittest.TestCase):
    def test_text_and_usage(self):
        t = OpenAiStreamTranslator()
        chunks = t.feed({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]})
        self.assertEqual([c.type for c in chunks], ["text-delta"])
        chunks = t.feed({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
        self.assertEqual(chunks, [MessageStop(TokenUsage(5, 2))])

    def test_reasoning_maps_to_reasoning_delta(self):
        t = OpenAiStreamTranslator()
        chunks = t.feed({"choices": [{"delta": {"reasoning_content": "hmm"}}]})
        self.assertEqual(chunks[0].type, "reasoning-delta")

    def test_tool_call_stream_opens_appends_closes(self):
        t = OpenAiStreamTranslator()
        c1 = t.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "echo", "arguments": ""}}
        ]}}]})
        self.assertEqual([c.type for c in c1], ["tool-call-start"])
        c2 = t.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"text"'}}
        ]}}]})
        self.assertEqual([c.type for c in c2], ["tool-call-delta"])
        c3 = t.feed({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        self.assertEqual([c.type for c in c3], ["tool-call-end"])

    def test_finish_guarantees_a_stop(self):
        t = OpenAiStreamTranslator()
        t.feed({"choices": [{"delta": {"content": "x"}}]})
        chunks = t.finish()
        self.assertEqual(chunks[-1], MessageStop())
        # Once stopped by usage, finish() adds nothing.
        t2 = OpenAiStreamTranslator()
        t2.feed({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        self.assertEqual(t2.finish(), [])


class FakeOpener:
    """Replays canned SSE lines and records the request it received."""

    def __init__(self, lines):
        self.lines = lines
        self.seen = None

    async def __call__(self, url, headers, body):
        self.seen = {"url": url, "headers": headers, "body": body}

        async def gen():
            for line in self.lines:
                yield line

        return gen()


class TestAdapterStream(unittest.IsolatedAsyncioTestCase):
    async def test_full_stream_round_trip(self):
        opener = FakeOpener([
            ": keep-alive",
            'data: {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}',
            'data: {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3}}',
            "data: [DONE]",
        ])
        adapter = DeepSeekAdapter("sk-test", opener=opener)
        request = ModelRequest(model="deepseek-chat", messages=(make_user_message("hi"),))
        chunks = [c async for c in adapter.stream(request)]

        self.assertEqual(
            [c.type for c in chunks],
            ["reasoning-delta", "text-delta", "text-delta", "message-stop"],
        )
        self.assertEqual(chunks[-1].usage.input_tokens, 9)
        # The opener saw the mapped request with auth.
        self.assertEqual(opener.seen["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(opener.seen["body"]["messages"][0]["content"], "hi")

    async def test_stream_ending_without_usage_still_stops(self):
        opener = FakeOpener([
            'data: {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}',
            "data: [DONE]",
        ])
        adapter = DeepSeekAdapter("sk-test", opener=opener)
        chunks = [c async for c in adapter.stream(ModelRequest(model="m", messages=()))]
        self.assertEqual(chunks[-1], MessageStop())

    async def test_empty_api_key_fails_loud(self):
        with self.assertRaises(AdapterError):
            DeepSeekAdapter("")

    async def test_read_sse_payloads_skips_non_data_lines(self):
        async def lines():
            yield ""
            yield ": ping"
            yield 'data: {"a": 1}'
            yield "data: [DONE]"
            yield 'data: {"never": "reached"}'

        payloads = [p async for p in read_sse_payloads(lines())]
        self.assertEqual(payloads, [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
