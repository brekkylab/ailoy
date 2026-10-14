"""An agent with a browser, waiting for what you ask of it.

    uv run main.py [--host]

The browser starts before the first request: Chromium in a VM of its own, which you watch and
use in yours at the address printed, opened on the ailoy docs. Each line you type is the next
turn; `/exit` ends. When a site needs you, to sign in say, the agent says so and ends its
turn: do it in the viewer, then tell it to go on.

With `--host` it uses your own Chrome instead, in a window of its own: turn on remote debugging
at chrome://inspect/#remote-debugging (Chrome 144 or later) and allow Chrome's prompt, which
appears as it starts.

Environment:

* `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
  API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), in the environment or
  in `.env`.

This folder is a uv project of its own (`pyproject.toml`), which `uv run` sets up with ailoy
from this checkout; run it from here.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# This folder is a uv project of its own. Started from another environment, run again in this one.
if Path(sys.prefix).resolve() != HERE / ".venv" and "AILOY_EXAMPLE_REEXEC" not in os.environ:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["AILOY_EXAMPLE_REEXEC"] = "1"  # once: not again if uv puts the environment elsewhere
    os.execvpe("uv", ["uv", "run", "--directory", str(HERE), "main.py", *sys.argv[1:]], env)

import ailoy  # noqa: E402
from ailoy.virtx import ConsoleClient, Recipe, ensure_virtx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Where the browser starts.
START = "https://brekkylab.github.io/ailoy/"

# What to ask, for a start.
SUGGESTIONS = [
    "Solve today's New York Times Wordle puzzle.",
    "Search Skyscanner and Google Flights for round-trip weekend flights from Seoul to Tokyo next month, and list the three cheapest by price.",
    "Find three wireless mice under $50 at an online store, and compare their prices and main features.",
]


def show(output: dict) -> None:
    message = output["message"]
    if message["role"] != "assistant":
        return
    for part in message["contents"]:
        if part["type"] == "text" and part["text"].strip():
            print(part["text"].strip(), flush=True)
    for call in message.get("tool_calls") or []:
        function = call["function"]
        print(f"  [{function['name']}] {json.dumps(function['arguments'], ensure_ascii=False)[:160]}", flush=True)


async def read(prompt: str) -> str | None:
    """A line from the terminal, or `None` once it closes."""
    try:
        return (await asyncio.to_thread(input, prompt)).strip()
    except EOFError:
        return None


async def main(model: str, host: bool) -> None:
    # The console server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none.
    await ensure_virtx()
    # The browser is driven from this console over the network, so it has to have one.
    console = await ConsoleClient.builder().image(Recipe("debian:trixie-slim")).network(True).build()
    builder = ailoy.AgentBuilder(model).system_tools().console(console)
    agent = await (builder.browser("host") if host else builder.browser()).build()
    print(f"model  {model}\n")
    try:
        if host:
            print("Allow Chrome's prompt to let the agent in.", flush=True)
        url = await agent.start_browser()
        print(f"Watch and use the browser at {url}\n" if url else "The agent has a window of its own in your Chrome.\n")

        # The app drives the agent's browser too, from the same console.
        print("Opening the ailoy docs as an example.\n", flush=True)
        opened = await console.exec(["agent-browser", "open", START])
        if opened.code != 0:
            print(f"Could not open {START}: {(opened.stdout + opened.stderr).decode(errors='replace').strip()}\n")

        print("Ask anything, or try:")
        for suggestion in SUGGESTIONS:
            print(f"  - {suggestion}")
        print("`/exit` ends.\n", flush=True)

        # A closed stdin ends the session, as `/exit` does.
        while (query := await read("> ")) not in (None, "/exit"):
            if query:
                async for output in agent.run(query):
                    show(output)
                print()
    finally:
        await agent.close()
        await console.close()


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    # Bytes that are not UTF-8, as a broken IME composition can leave, become U+FFFD.
    sys.stdin.reconfigure(errors="replace")
    asyncio.run(main(os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5"), "--host" in sys.argv[1:]))
