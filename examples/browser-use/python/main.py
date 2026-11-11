"""An agent with a browser, waiting for what you ask of it.

    uv run main.py [--noshell] [--isolated]

The agent uses your own browser (Chrome by default), in a window of its own opened on the ailoy
docs, and drives it with its `browser_*` tools. It also has a sandbox with python and curl, for
what is easier done there, such as fetching a page's data or computing over it; `--noshell` leaves
it out, so the agent has the browser alone and needs no sandbox server. Turn on remote debugging at
chrome://inspect/#remote-debugging (Chrome 144 or later) and allow the browser's prompt, which
appears as it starts. `--isolated` gives the agent a browser of its own instead: a Chromium-based
one installed here, started with a fresh profile and closed at the end, with no remote debugging
to turn on. Each line you type is the next turn; `/exit` ends. When a site needs you, to
sign in say, the agent says so and ends its turn: do it in its window, then tell it to go on.

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
from virtx import Client, ImageJobEnd, Recipe  # noqa: E402

# The image the sandbox boots, built once on the virtx server and kept under this tag.
IMAGE = "ailoy-browser-use:1"

# Where the browser starts.
START = "https://ailoy.co/docs/"

# What the agent has, with a sandbox and without.
WITH_SANDBOX = "a web browser and a sandbox: the browser is {whose}, which you drive with the browser tools"
BROWSER_ONLY = "a web browser: {whose}, which you drive with the browser tools"

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


async def main(model: str, shell: bool, isolated: bool) -> None:
    # The browser tools reach the browser from this process; the sandbox is for the other tools.
    whose = "your own, apart from the user's" if isolated else "the user's own"
    has = (WITH_SANDBOX if shell else BROWSER_ONLY).format(whose=whose)
    builder = ailoy.AgentBuilder(model).instruction(
        f"You are a helpful assistant with {has}. {INSTRUCTION}"
    )
    builder = builder.browser(start_url=START, isolated=isolated)
    client = None
    if shell:
        # The virtx server, fetched into virtx's cache the first time: a host that installed
        # only ailoy has none. The sandboxes it holds go when this program does.
        client = Client.spawn()

        # The image, built once: python, and curl on it.
        # The server answers a tag in its full form, as `docker.io/library/ailoy-browser-use:1`.
        images = await client.list_image()
        if not any(image.reference.split("/")[-1] == IMAGE for image in images):
            print(f"Building {IMAGE}, once; it takes a few minutes.")
            recipe = Recipe("python:3.12-slim-trixie").run(
                "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*"
            )
            async for entry in client.build_image(recipe, IMAGE):
                if isinstance(entry.entry, ImageJobEnd) and entry.entry.error:
                    raise RuntimeError(f"building {IMAGE}: {entry.entry.error}")

        sandbox = await client.create_sandbox(IMAGE, network=True)
        builder = builder.sandbox(sandbox)
    agent = await builder.build()
    print(f"model     {model}")
    print(f"has       {'browser, sandbox' if shell else 'browser'}\n")
    try:
        if isolated:
            print(
                "Starting the agent's own browser. It opens the ailoy docs as an example.",
                flush=True,
            )
        else:
            print(
                "Allow the browser's prompt to let the agent in. It opens the ailoy docs as an example.",
                flush=True,
            )
        await agent.start_browser()
        print(
            "The agent's browser is open.\n"
            if isolated
            else "The agent has a window of its own in your browser.\n"
        )

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
        if client:
            client.close()


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    # Bytes that are not UTF-8, as a broken IME composition can leave, become U+FFFD.
    sys.stdin.reconfigure(errors="replace")
    args = sys.argv[1:]
    asyncio.run(
        main(
            os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5"),
            "--noshell" not in args,
            "--isolated" in args,
        )
    )
