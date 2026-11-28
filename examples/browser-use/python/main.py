"""An agent with a browser, waiting for what you ask of it.

    uv run main.py

The agent works in a sandbox with python and curl and virtx's `browser` abin, a headless
Chromium, which it drives with its `browser` and `browser_screenshot` tools; the shell and file
tools are there too, for what is easier done without the browser, such as fetching a page's data
or computing over it. Each line you type is the next turn; `/exit` ends.

Environment:

* `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend has
  to be set up as `docs/content/guides/model-providers.mdx` says, any key in the
  environment or in `.env`.

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
if (
    Path(sys.prefix).resolve() != HERE / ".venv"
    and "AILOY_EXAMPLE_REEXEC" not in os.environ
):
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["AILOY_EXAMPLE_REEXEC"] = (
        "1"  # once: not again if uv puts the environment elsewhere
    )
    os.execvpe(
        "uv", ["uv", "run", "--directory", str(HERE), "main.py", *sys.argv[1:]], env
    )

import ailoy  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from virtx import Client, Recipe  # noqa: E402

# The image the sandbox boots, built once on the virtx server and kept under this tag.
IMAGE = "ailoy-browser-use:1"

# What the agent has.
HAS = "a sandbox and a web browser in it, which you drive with the browser tools"

# What the agent is for.
INSTRUCTION = (
    "Do what the user asks, and say briefly what you are doing as you go. When you are done, "
    "report what you found plainly, with the figures and links that matter. Answer in the user's language."
)

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
        print(
            f"  [{function['name']}] {json.dumps(function['arguments'], ensure_ascii=False)[:160]}",
            flush=True,
        )


async def read(prompt: str) -> str | None:
    """A line from the terminal, or `None` once it closes."""
    try:
        return (await asyncio.to_thread(input, prompt)).strip()
    except EOFError:
        return None


async def main(model: str) -> None:
    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()
    try:
        # The image, built once: python, and curl on it.
        # The server answers a tag in its full form, as `docker.io/library/ailoy-browser-use:1`.
        images = await client.list_image()
        if not any(image.reference.split("/")[-1] == IMAGE for image in images):
            print(f"Building {IMAGE}, once; it takes a few minutes.")
            recipe = Recipe("python:3.12-slim-trixie").run(
                "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*"
            )
            await client.build_image(recipe, IMAGE).complete()

        # The browser is the sandbox's: its abin, a network for the pages it opens, and the
        # memory Chromium takes.
        sandbox = await client.create_sandbox(
            IMAGE, network=True, abins=["browser"], memory_mib=2048
        )
        agent = await (
            ailoy.AgentBuilder(model)
            .instruction(f"You are a helpful assistant with {HAS}. {INSTRUCTION}")
            .sandbox(sandbox)
            .browser()
            .build()
        )
        print(f"model     {model}\n")
        try:
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
    finally:
        client.close()


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    # Bytes that are not UTF-8, as a broken IME composition can leave, become U+FFFD.
    sys.stdin.reconfigure(errors="replace")
    asyncio.run(main(os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5")))
