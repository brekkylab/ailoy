"""Run one agent turn with a fixed question: no tools, no console, no system message.

    uv run main.py [model]

`model` defaults to `bedrock/global.openai.gpt-5.6-luna`; its provider's API key has to be
set (`AWS_BEARER_TOKEN_BEDROCK`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), in the
environment or in `.env`.

This folder is a uv project of its own (`pyproject.toml`), which `uv run` sets up with
ailoy from this checkout; run it from here.
"""

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# This folder is a uv project of its own. Started from another environment, as with
# `uv run python examples/hello/main.py` from `bindings/python`, run again in this one.
if Path(sys.prefix).resolve() != HERE / ".venv" and "AILOY_EXAMPLE_REEXEC" not in os.environ:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["AILOY_EXAMPLE_REEXEC"] = "1"  # once: not again if uv puts the environment elsewhere
    os.execvpe("uv", ["uv", "run", "--directory", str(HERE), "main.py", *sys.argv[1:]], env)

import ailoy  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# The request the agent is given.
QUERY = "What is the meaning of hello world?"


async def main(model: str) -> None:
    agent = await ailoy.AgentBuilder(model).build()
    print(f"model  {model}\n")

    async for output in agent.run(QUERY):
        message = output["message"]
        if message["role"] == "assistant":
            for part in message["contents"]:
                if part["type"] == "text":
                    print(part["text"], flush=True)


if __name__ == "__main__":
    # From the nearest `.env` up from this file, as the Rust examples load it.
    load_dotenv()
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "bedrock/global.openai.gpt-5.6-luna"))
