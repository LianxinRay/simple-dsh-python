"""Approval demo: the model must ask before touching the filesystem.

Gates ``write_file``/``edit_file``/``bash`` behind console approval — each
gated call pauses with a y/N prompt; denying makes the tool call fail and
the model must cope. Self-skips without DEEPSEEK_API_KEY.

    python examples/approval_demo.py            # interactive y/N
    echo y | python examples/approval_demo.py   # auto-approve (CI-style)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_dsh.app import create_agent, create_app
from simple_dsh.approval import console_responder
from simple_dsh.credentials import Credentials
from simple_dsh.llm import TextDelta


async def main() -> None:
    here = Path(__file__).resolve().parent.parent
    env_path = here / ".env"
    credentials = Credentials(env_path if env_path.is_file() else None)
    if not credentials.get("DEEPSEEK_API_KEY"):
        print("SKIP: DEEPSEEK_API_KEY not found in environment or .env")
        return

    workspace = here / "examples" / "out" / "approval-workspace"
    ctx = create_app(
        workspace,
        env_path=env_path,
        preset={
            "approval": {"tools": ["write_file", "edit_file", "bash"]},
        },
        approval_responder=console_responder,
        log_path=workspace / "session.jsonl",
    )
    agent = create_agent(ctx)
    ctx.on(
        "agent/chunk",
        lambda c: print(c.text, end="", flush=True) if isinstance(c, TextDelta) else None,
    )

    print("Model will try to write a file — approve or deny each gated call.\n")
    await agent.prompt("Create a file approved.txt containing 'approved by human'.")
    await agent.when_idle()

    target = workspace / "approved.txt"
    print(f"\n\nfile exists: {target.exists()}")
    gated = [e for e in agent.session if e.type == "tool/call"]
    results = [e for e in agent.session if e.type == "tool/result"]
    print(f"tool calls: {len(gated)}, results: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
