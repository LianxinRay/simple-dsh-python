"""Tests for the LLM vocabulary and stream assembler."""

import unittest

from simple_dsh.llm import (
    AssistantProvenance,
    MessageStop,
    ReasoningDelta,
    StreamAssembler,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    block_from_json,
    block_to_json,
    chunk_from_json,
    chunk_to_json,
)

PROV = AssistantProvenance(provider="test", model="m1")


class TestAssembler(unittest.TestCase):
    def test_text_deltas_accumulate_into_one_block(self):
        asm = StreamAssembler(PROV)
        asm.feed(TextDelta("hello"))
        asm.feed(TextDelta(" world"))
        asm.feed(MessageStop(TokenUsage(3, 2)))
        message = asm.finish()
        self.assertEqual(message.content, (TextBlock("hello world"),))
        self.assertEqual(message.source.provenance, PROV)
        self.assertEqual(asm.usage, TokenUsage(3, 2))

    def test_reasoning_is_a_distinct_block(self):
        asm = StreamAssembler(PROV)
        asm.feed(ReasoningDelta("thinking..."))
        asm.feed(TextDelta("answer"))
        asm.feed(MessageStop())
        kinds = [b.type for b in asm.finish().content]
        self.assertEqual(kinds, ["reasoning", "text"])

    def test_tool_call_assembled_from_chunks(self):
        asm = StreamAssembler(PROV)
        asm.feed(ToolCallStart(id="c1", name="echo"))
        asm.feed(ToolCallDelta(id="c1", arguments_delta='{"text":'))
        asm.feed(ToolCallDelta(id="c1", arguments_delta='"hi"}'))
        asm.feed(ToolCallEnd(id="c1"))
        asm.feed(MessageStop())
        (call,) = [b for b in asm.finish().content if isinstance(b, ToolCallBlock)]
        self.assertEqual((call.id, call.name, call.arguments), ("c1", "echo", '{"text":"hi"}'))

    def test_finish_requires_message_stop(self):
        asm = StreamAssembler(PROV)
        asm.feed(TextDelta("partial"))
        with self.assertRaises(ValueError):
            asm.finish()
        # Interrupted finalizes the delivered prefix instead.
        self.assertEqual(asm.finish(interrupted=True).content, (TextBlock("partial"),))


class TestJsonRoundTrip(unittest.TestCase):
    def test_block_round_trip(self):
        for block in [
            TextBlock("hi"),
            ToolCallBlock(id="c", name="t", arguments="{}"),
        ]:
            self.assertEqual(block_from_json(block_to_json(block)), block)

    def test_chunk_round_trip(self):
        for chunk in [
            TextDelta("x"),
            ToolCallStart(id="c", name="t"),
            MessageStop(TokenUsage(1, 2)),
            MessageStop(),
        ]:
            self.assertEqual(chunk_from_json(chunk_to_json(chunk)), chunk)


if __name__ == "__main__":
    unittest.main()
