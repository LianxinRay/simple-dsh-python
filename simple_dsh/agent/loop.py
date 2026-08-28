"""The turn/step state machine that drives one agent."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Literal

from ..cordis import Context
from ..llm import (
    AssistantProvenance,
    LlmRegistry,
    Message,
    ModelRequest,
    StreamAssembler,
    ToolCallBlock,
    chunk_to_json,
    message_to_json,
)
from ..session import Session, TurnEndReason, make_tool_result_message, make_user_message
from ..tools import ToolRegistry

AgentStatus = Literal["idle", "running", "cancelled"]


class _Reject:
    def __repr__(self) -> str:
        return "REJECT"


REJECT = _Reject()
"""Returned from an ``agent/pre-step`` waterfall listener to reject the claim."""


@dataclass(frozen=True)
class _InboxItem:
    """Queued input. ``kind='prompt'`` wakes the driver; ``'inject'`` waits."""

    kind: Literal["prompt", "inject"]
    message: Message


class Agent:
    """One live agent: an inbox, a session log, and the loop that drains them.

    Injected context (``inject()``) waits in the inbox until a real prompt
    wakes the driver — matching the upstream inbox contract. Every
    model-visible fact is appended to the session log *before* history is
    derived for the next request, preserving the model-visible-means-logged
    invariant.
    """

    def __init__(
        self,
        ctx: Context,
        session: Session | None = None,
        model: str = "default",
    ) -> None:
        self.ctx = ctx.scope("agent")
        self.session = session if session is not None else Session()
        self.model = model
        self._inbox: asyncio.Queue[_InboxItem] = asyncio.Queue()
        self._driver: asyncio.Task[None] | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        self.status: AgentStatus = "idle"

    # ------------------------------------------------------------------ API

    async def prompt(self, text: str) -> None:
        """Queue a human prompt and wake the driver."""
        await self._inbox.put(_InboxItem("prompt", make_user_message(text)))
        self._ensure_driver()

    def inject(self, text: str, **extra: object) -> None:
        """Queue synthetic context; it lands in the next admitted request."""
        self._inbox.put_nowait(
            _InboxItem("inject", make_user_message(text, source_kind="plugin", **extra))
        )

    def cancel(self) -> None:
        """Cancel the in-flight turn."""
        self.status = "cancelled"
        if self._driver is not None:
            self._driver.cancel()

    async def when_idle(self) -> None:
        """Wait until no turn is running and the inbox is drained."""
        await self._idle.wait()
        if self._driver is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._driver

    # ---------------------------------------------------------------- driver

    def _ensure_driver(self) -> None:
        if self._driver is None or self._driver.done():
            self._idle.clear()
            self._driver = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while not self._inbox.empty():
                await self._run_turn()
        finally:
            self._idle.set()
            if self.status != "cancelled":
                self.status = "idle"

    def _claim(self) -> list[Message]:
        """Claim one queued prompt plus every waiting injection.

        The prompt leads the batch regardless of queue order; extra prompts
        stay queued, in order, for their own turns.
        """
        items: list[_InboxItem] = []
        while not self._inbox.empty():
            items.append(self._inbox.get_nowait())
        claimed: list[Message] = []
        injects: list[Message] = []
        prompt_taken = False
        for item in items:
            if item.kind == "prompt":
                if prompt_taken:
                    self._inbox.put_nowait(item)
                else:
                    claimed.append(item.message)
                    prompt_taken = True
            else:
                injects.append(item.message)
        return claimed + injects

    # ------------------------------------------------------------------ turn

    async def _run_turn(self) -> None:
        session = self.session
        turn = self._next_turn_number()
        session.append("turn/start", {"turn": turn})
        reason: TurnEndReason = "completed"
        self.status = "running"
        step = 0
        try:
            claimed = self._claim()
            admitted = await self.ctx.waterfall("agent/pre-step", claimed)
            # A rejected or empty first claim closes the turn with no step.
            if not claimed:
                reason = "empty"
            elif admitted is REJECT:
                reason = "rejected"
            elif not admitted:
                reason = "empty"
            else:
                while True:
                    step += 1
                    made_calls = await self._run_step(turn, step, admitted)
                    if not made_calls:
                        break
                    # Tools owe another request. Newly queued input joins the
                    # continuation; pre-step may still reject it.
                    admitted = await self.ctx.waterfall("agent/pre-step", self._claim())
                    if admitted is REJECT:
                        break
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except Exception:
            reason = "error"
            raise
        finally:
            session.append("turn/end", {"turn": turn, "reason": reason})

    # ------------------------------------------------------------------ step

    async def _run_step(self, turn: int, step: int, admitted: list[Message]) -> bool:
        """Run one model request plus its tools. Returns True when the model
        made tool calls, meaning the step owes another request."""
        session = self.session
        session.append("step/start", {"turn": turn, "step": step})
        for message in admitted:
            session.append("user/message", message_to_json(message))

        # History is derived from the log only after this step's own input
        # has been appended — model-visible means logged.
        history = tuple(session.derive_messages())
        system = await self.ctx.systemPrompt.assemble()
        request = ModelRequest(
            model=self.model,
            messages=history,
            system=system,
            tools=self._tools().schemas(),
        )

        adapter = self._llm().resolve(self.model)
        assembler = StreamAssembler(
            AssistantProvenance(provider=adapter.provider, model=self.model)
        )
        async for chunk in adapter.stream(request):
            session.append(
                "assistant/chunk", {"turn": turn, "step": step, "chunk": chunk_to_json(chunk)}
            )
            assembler.feed(chunk)
            await self.ctx.emit("agent/chunk", chunk)

        message = assembler.finish()
        usage = (
            {
                "input_tokens": assembler.usage.input_tokens,
                "output_tokens": assembler.usage.output_tokens,
            }
            if assembler.usage
            else None
        )
        session.append(
            "assistant/message",
            {
                "turn": turn,
                "step": step,
                "message": message_to_json(message),
                **({"usage": usage} if usage else {}),
            },
        )

        calls = [b for b in message.content if isinstance(b, ToolCallBlock)]
        for call in calls:
            session.append(
                "tool/call",
                {
                    "turn": turn,
                    "step": step,
                    "callId": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                },
            )
            result = await self._tools().execute(call)
            session.append(
                "tool/result",
                {
                    "turn": turn,
                    "step": step,
                    "message": message_to_json(
                        make_tool_result_message(call.id, result.content, result.is_error)
                    ),
                    **({"meta": result.meta} if result.meta is not None else {}),
                },
            )
        session.append("step/end", {"turn": turn, "step": step})
        return bool(calls)

    # ---------------------------------------------------------------- helpers

    def _next_turn_number(self) -> int:
        """Next 1-based turn number, derived from the log."""
        return sum(1 for e in self.session if e.type == "turn/start") + 1

    def _llm(self) -> LlmRegistry:
        return self.ctx.service("llm")

    def _tools(self) -> ToolRegistry:
        return self.ctx.service("tools")
