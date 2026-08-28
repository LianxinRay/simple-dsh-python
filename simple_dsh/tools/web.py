"""Web fetch tool (P2).

A minimal ``packages/web`` analogue: one model-facing ``web_fetch`` tool
(stdlib transport) that returns a page's visible text, truncated. Search
needs a provider API and is deliberately not hand-rolled here.
"""

from __future__ import annotations

import asyncio
import urllib.request
from html.parser import HTMLParser

from ..cordis import Disposer
from ..llm import TextBlock
from .registry import ToolDefinition, ToolRegistry, ToolResult

_MAX_CHARS = 20_000


class _TextExtractor(HTMLParser):
    """Collects visible text, skipping script/style blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    """Extract visible text from an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    return "\n".join(parser.parts)


def register_web_tool(tools: ToolRegistry, *, name: str = "web_fetch") -> Disposer:
    """Register the fetch tool; returns its disposer."""

    async def web_fetch(args):
        url = args["url"]
        if not url.startswith(("http://", "https://")):
            return ToolResult.text("url must start with http:// or https://", is_error=True)

        def _fetch() -> str:
            request = urllib.request.Request(url, headers={"User-Agent": "simple-dsh/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content_type = response.headers.get("content-type", "")
                body = response.read(2_000_000).decode("utf-8", errors="replace")
            return html_to_text(body) if "html" in content_type else body

        text = await asyncio.to_thread(_fetch)
        truncated = len(text) > _MAX_CHARS
        if truncated:
            text = text[:_MAX_CHARS] + "\n... [truncated]"
        return ToolResult(
            content=[TextBlock(text=text or "(empty page)")],
            meta={"truncated": truncated} if truncated else None,
        )

    return tools.register(ToolDefinition(
        name=name,
        description="Fetch a URL and return its visible text (truncated).",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        execute=web_fetch,
    ))


__all__ = ["html_to_text", "register_web_tool"]
