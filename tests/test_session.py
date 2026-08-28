"""Tests for the session event log."""

import tempfile
import unittest
from pathlib import Path

from simple_dsh.llm import TextBlock, message_from_json, message_to_json
from simple_dsh.session import (
    JsonlSink,
    Session,
    load_jsonl,
    make_tool_result_message,
    make_user_message,
)


class TestSessionLog(unittest.TestCase):
    def test_append_assigns_contiguous_seq(self):
        session = Session()
        e0 = session.append("turn/start", {"turn": 1})
        e1 = session.append("turn/end", {"turn": 1, "reason": "completed"})
        self.assertEqual((e0.seq, e1.seq), (0, 1))
        self.assertEqual(session.at(1).type, "turn/end")

    def test_non_json_payload_rejected_at_source(self):
        session = Session()
        with self.assertRaises(TypeError):
            session.append("user/message", {"bad": object()})

    def test_derive_messages_projects_in_log_order(self):
        session = Session()
        user = make_user_message("question")
        assistant_msg = make_user_message("answer")  # role swapped below
        import dataclasses

        assistant_msg = dataclasses.replace(assistant_msg, role="assistant")
        tool_msg = make_tool_result_message("c1", [TextBlock("tool output")], False)
        session.append("user/message", message_to_json(user))
        session.append("assistant/message", {"turn": 1, "step": 1, "message": message_to_json(assistant_msg)})
        session.append("tool/result", {"turn": 1, "step": 1, "message": message_to_json(tool_msg)})
        derived = session.derive_messages()
        self.assertEqual([m.role for m in derived], ["user", "assistant", "user"])
        self.assertEqual(derived[2].content[0].type, "tool-result")

    def test_chunks_never_enter_derived_history(self):
        session = Session()
        session.append("assistant/chunk", {"turn": 1, "step": 1, "chunk": {"type": "text-delta", "text": "x"}})
        self.assertEqual(session.derive_messages(), [])

    def test_seeded_events_need_contiguous_seq(self):
        with self.assertRaises(ValueError):
            from simple_dsh.session.events import SessionEvent

            Session([SessionEvent(seq=7, type="turn/start", data={"turn": 1})])


class TestJsonlPersistence(unittest.TestCase):
    def test_round_trip_preserves_log_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            session = Session()
            sink = JsonlSink(path)
            session.add_sink(sink)
            session.append("user/message", message_to_json(make_user_message("hello")))
            session.append("turn/end", {"turn": 1, "reason": "completed"})
            sink.close()

            events = load_jsonl(path)
            self.assertEqual([e.type for e in events], ["user/message", "turn/end"])
            restored = Session(events)
            derived = restored.derive_messages()
            self.assertEqual(len(derived), 1)
            self.assertEqual(derived[0].content[0], TextBlock("hello"))

    def test_message_json_round_trip(self):
        message = make_user_message("hi", source_kind="plugin", origin="test")
        self.assertEqual(message_from_json(message_to_json(message)), message)


if __name__ == "__main__":
    unittest.main()
