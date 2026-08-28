"""Real-API demo: run the agent against DeepSeek, end to end.

Reads ``DEEPSEEK_API_KEY`` env-over-.env (checked at the repo root of the
sibling ``deepseek-harness`` checkout or ``./.env``). Self-skips without a
key, mirroring the upstream e2e policy:

    python examples/deepseek_demo.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_dsh.app import create_agent, create_app
from simple_dsh.credentials import Credentials
from simple_dsh.llm import TextBlock


async def main() -> None:
    here = Path(__file__).resolve().parent.parent
    candidates = [here / ".env", here.parent / "deepseek-harness" / ".env"]
    env_path = next((p for p in candidates if p.is_file()), None)
    credentials = Credentials(env_path)
    if not credentials.get("DEEPSEEK_API_KEY"):
        print("SKIP: DEEPSEEK_API_KEY not found in environment or .env")
        return

    workspace = here / "examples" / "out" / "workspace"
    ctx = create_app(workspace, env_path=env_path, log_path=workspace / "session.jsonl")
    agent = create_agent(ctx)

    async def show(chunk):
        if isinstance(chunk, TextBlock):
            print(chunk.text, end="", flush=True)

    # Observe the live stream; the durable log holds the authoritative record.
    from simple_dsh.llm import TextDelta
    ctx.on("agent/chunk", lambda c: show(c) if isinstance(c, TextDelta) else None)

    await agent.prompt(
        "Create a file hello.txt containing 'hello from deepseek', "
        "then read it back and tell me what it says."
    )
    await agent.when_idle()

    print("\n\n=== session log ===")
    for event in agent.session:
        print(f"  seq={event.seq:<2} {event.type}")


if __name__ == "__main__":
    asyncio.run(main())
