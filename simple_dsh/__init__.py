"""simple-dsh-python: a minimal Python port of the DeepSeek Harness P0 spine.

Layers, in dependency order:

- :mod:`simple_dsh.cordis` — plugin runtime: Context, services, typed events,
  effects.
- :mod:`simple_dsh.llm` — conversation vocabulary: messages, content blocks,
  stream chunks, the adapter seam, the assembler.
- :mod:`simple_dsh.session` — the append-only session event log and derived
  model history, with JSONL durability.
- :mod:`simple_dsh.prompts` — system-prompt section registry and assembly.
- :mod:`simple_dsh.tools` — the tool registry and guarded execution pipeline.
- :mod:`simple_dsh.agent` — the Agent handle and the turn/step loop driver.
"""

from . import agent, approval, compaction, cordis, llm, preset, prompts, session, tools

__all__ = [
    "agent", "approval", "compaction", "cordis", "llm", "preset",
    "prompts", "session", "tools",
]
__version__ = "0.2.0"
