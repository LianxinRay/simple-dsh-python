"""The Agent handle and the default loop driver (P0-6).

Ported from ``packages/core/agent`` + ``packages/core/agent-loop``: a *step*
is one model request plus the tools it calls; a *turn* is zero or more steps.
The driver claims queued input, runs the ``agent/pre-step`` waterfall,
appends every model-visible fact to the session log, streams the model
response, dispatches tool calls through ``ctx.tools``, and loops until
nothing is owed.
"""

from .loop import REJECT, Agent, AgentStatus

__all__ = ["REJECT", "Agent", "AgentStatus"]
