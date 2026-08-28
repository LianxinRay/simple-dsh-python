"""Tool registry and guarded execution pipeline (P0-5).

Ported from ``packages/core/tools``: a tool is a name, a description, a JSON
parameter schema, and an async ``execute`` body. Every call flows through
the ``tools/pre-execute`` waterfall (policy may deny), registered monotonic
guards (deny or abstain), the ``tools/execute`` around-waterfall (timeouts,
metrics) with the tool body at its terminal, and the ``tools/post-execute``
waterfall (may replace or annotate the result). Any pipeline exception is
normalized into an ``is_error`` result — the loop never sees a raise.
"""

from .registry import ToolDefinition, ToolRegistry, ToolResult

__all__ = ["ToolDefinition", "ToolRegistry", "ToolResult"]
