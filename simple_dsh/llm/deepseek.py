"""DeepSeek streaming adapter (P1).

Ported from ``packages/llm/llm-deepseek``: speaks the OpenAI-compatible
chat-completions SSE protocol and translates provider chunks into the shared
``StreamChunk`` vocabulary. Transport is stdlib-only
(``asyncio.open_connection`` + TLS + chunked decoding); the opener is
injectable so tests replay canned streams without a network.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import urllib.parse
from typing import Any, AsyncIterator, Awaitable, Callable

from .types import (
    ContentBlock,
    ImageBlock,
    Message,
    MessageStop,
    ModelRequest,
    ReasoningBlock,
    ReasoningDelta,
    StreamChunk,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
)


class AdapterError(RuntimeError):
    """A provider request failed (HTTP error or malformed stream)."""


# --------------------------------------------------------- request building


def _message_to_openai(message: Message) -> list[dict[str, Any]]:
    """Project one internal message into OpenAI wire messages.

    Tool-result blocks become ``role: tool`` messages; reasoning blocks are
    dropped (providers do not accept them back); images are unsupported by
    this minimal adapter and fail loud.
    """
    out: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ImageBlock):
            raise AdapterError("image blocks are not supported by the DeepSeek adapter yet")
    if message.role == "assistant":
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        wire: dict[str, Any] = {"role": "assistant", "content": text or None}
        tool_calls = [b for b in message.content if isinstance(b, ToolCallBlock)]
        if tool_calls:
            wire["tool_calls"] = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": b.arguments},
                }
                for b in tool_calls
            ]
        out.append(wire)
        return out
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            content = "".join(b.text for b in block.content if isinstance(b, TextBlock))
            out.append(
                {"role": "tool", "tool_call_id": block.tool_call_id, "content": content}
            )
        elif isinstance(block, TextBlock):
            out.append({"role": message.role, "content": block.text})
        elif isinstance(block, ReasoningBlock):
            continue
    return out


def request_to_openai(request: ModelRequest) -> dict[str, Any]:
    """Build the OpenAI-compatible request body for one model request."""
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for message in request.messages:
        messages.extend(_message_to_openai(message))
    body: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
            for schema in request.tools
        ]
    return body


# ------------------------------------------------------ stream translation


class OpenAiStreamTranslator:
    """Folds provider payload dicts into ``StreamChunk``\\ s.

    Stateful on streamed tool calls: the first delta for a call index opens
    it, argument fragments append, and the call closes when the next call
    starts or the message finishes.
    """

    def __init__(self) -> None:
        self._open: dict[int, str] = {}  # index -> call id
        self._stopped = False

    def feed(self, payload: dict[str, Any]) -> list[StreamChunk]:
        """Translate one SSE payload; may yield zero or more chunks."""
        chunks: list[StreamChunk] = []
        choices = payload.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if delta.get("reasoning_content"):
                chunks.append(ReasoningDelta(thinking=delta["reasoning_content"]))
            if delta.get("content"):
                chunks.append(TextDelta(text=delta["content"]))
            for tool_delta in delta.get("tool_calls") or []:
                chunks.extend(self._feed_tool_call(tool_delta))
            if choices[0].get("finish_reason") is not None:
                chunks.extend(self._close_all())
        usage = payload.get("usage")
        if usage and not self._stopped:
            self._stopped = True
            chunks.append(
                MessageStop(
                    TokenUsage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    )
                )
            )
        return chunks

    def finish(self) -> list[StreamChunk]:
        """Close the stream: flush open tool calls, guarantee one stop chunk.

        Called when the SSE stream ends. Providers that send a usage chunk
        have already produced their ``MessageStop``; for the rest this emits
        one without usage so the assembler always sees a terminated message.
        """
        chunks = self._close_all()
        if not self._stopped:
            self._stopped = True
            chunks.append(MessageStop())
        return chunks

    def _feed_tool_call(self, tool_delta: dict[str, Any]) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        index = tool_delta.get("index", 0)
        call_id = tool_delta.get("id")
        function = tool_delta.get("function") or {}
        if call_id is not None:
            # A new call at this index closes any call still open there.
            for open_index in [i for i in self._open if i < index]:
                chunks.append(ToolCallEnd(id=self._open.pop(open_index)))
            if index in self._open:
                chunks.append(ToolCallEnd(id=self._open.pop(index)))
            self._open[index] = call_id
            chunks.append(ToolCallStart(id=call_id, name=function.get("name", "")))
        if function.get("arguments"):
            chunks.append(
                ToolCallDelta(id=self._open[index], arguments_delta=function["arguments"])
            )
        return chunks

    def _close_all(self) -> list[StreamChunk]:
        chunks = [ToolCallEnd(id=call_id) for _, call_id in sorted(self._open.items())]
        self._open.clear()
        return chunks


# ------------------------------------------------------------ SSE transport


async def read_sse_payloads(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Parse an SSE line stream into payload dicts until ``[DONE]``."""
    async for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


async def _chunked_lines(reader: asyncio.StreamReader) -> AsyncIterator[str]:
    """Decode an HTTP chunked body into text lines."""
    buffer = b""
    while True:
        size_line = (await reader.readline()).strip()
        if not size_line:
            continue
        size = int(size_line, 16)
        if size == 0:
            await reader.readline()  # trailing CRLF
            return
        buffer += await reader.readexactly(size)
        await reader.readexactly(2)  # the CRLF terminating each chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            yield raw.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ adapter

# (url, headers, body) -> async iterator of decoded body lines
StreamOpener = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[AsyncIterator[str]]]


class DeepSeekAdapter:
    """``ctx.llm`` adapter for DeepSeek's OpenAI-compatible streaming API."""

    provider = "deepseek"
    models = ("deepseek-chat", "deepseek-reasoner")

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        opener: StreamOpener | None = None,
    ) -> None:
        if not api_key:
            raise AdapterError("DeepSeekAdapter requires a non-empty api_key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._opener = opener or self._open_stream

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        body = request_to_openai(request)
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        lines = await self._opener(f"{self._base_url}/chat/completions", headers, body)
        translator = OpenAiStreamTranslator()
        async for payload in read_sse_payloads(lines):
            for chunk in translator.feed(payload):
                yield chunk
        for chunk in translator.finish():
            yield chunk

    # -------------------------------------------------------- stdlib opener

    async def _open_stream(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> AsyncIterator[str]:
        """POST ``body`` as JSON and yield the chunked SSE body line by line."""
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        tls = ssl.create_default_context() if parsed.scheme == "https" else None
        reader, writer = await asyncio.open_connection(host, port, ssl=tls)
        payload = json.dumps(body).encode("utf-8")
        path = parsed.path or "/"
        request_head = (
            f"POST {path} HTTP/1.1\r\nhost: {host}\r\n"
            + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            + f"content-length: {len(payload)}\r\nconnection: close\r\n\r\n"
        )
        writer.write(request_head.encode("ascii") + payload)
        await writer.drain()

        status_line = (await reader.readline()).decode("ascii", errors="replace")
        try:
            status = int(status_line.split()[1])
        except (IndexError, ValueError) as exc:
            writer.close()
            raise AdapterError(f"malformed HTTP status line: {status_line!r}") from exc
        response_headers: dict[str, str] = {}
        while True:
            line = (await reader.readline()).decode("ascii", errors="replace")
            if line in ("\r\n", ""):
                break
            key, _, value = line.partition(":")
            response_headers[key.strip().lower()] = value.strip()
        if status != 200:
            error_body = (await reader.read()).decode("utf-8", errors="replace")
            writer.close()
            raise AdapterError(f"HTTP {status} from DeepSeek: {error_body[:500]}")
        if "chunked" not in response_headers.get("transfer-encoding", ""):
            writer.close()
            raise AdapterError("expected a chunked SSE response")

        return _close_on_done(_chunked_lines(reader), writer)


async def _close_on_done(
    lines: AsyncIterator[str], writer: asyncio.StreamWriter
) -> AsyncIterator[str]:
    try:
        async for line in lines:
            yield line
    finally:
        writer.close()


__all__ = [
    "AdapterError",
    "DeepSeekAdapter",
    "OpenAiStreamTranslator",
    "read_sse_payloads",
    "request_to_openai",
]
