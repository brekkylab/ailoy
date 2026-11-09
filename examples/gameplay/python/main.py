"""An agent that plays OpenTTD, running in a sandbox, while you watch it over VNC.

    uv run main.py
    uv run main.py "Connect every town over 1000 people by air"

The Python side of the Rust `gameplay` example in `examples/gameplay/rust`, whose `main.rs` has
the long form. The game runs as a dedicated server in a sandbox, and the agent runs a company
in it from the same sandbox, through the skill's `ttd.py`. `examples/gameplay/shared`, the
skill (`SKILL.md`, `ttd.py`, `admin.py`) and the game's side (`server/`: `start.sh`, the Game
Script and the AI) that the three sides share, goes each run onto a volume of the virtx server
mounted at `/skills/openttd`. This folder is a uv project of its own
(`pyproject.toml`) that `uv run` sets up with ailoy from this checkout; run it from here.

* The volume `ailoy-gameplay-artifacts` at `/artifacts`, writable — the agent's notes, the
  screenshots it took, the saved games in `user/openttd/save/` (the last as
  `ailoy-final.sav`), and the logs. It stays on the server between runs, and is copied to
  `artifacts/` here at the end.

Watch at the `vnc://` address it prints (Screen Sharing on macOS, or any VNC viewer) with the
password `openttd`.

Environment:

* `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend has
  to be set up as `docs/src/content/docs/guide/model-providers.mdx` says. It reads
  screenshots when it looks, so it should take images.
* `OPENTTD_YEARS` — how many years of the game the agent plays, `3` by default. A game month
  takes a minute to pass.
* `OPENTTD_SAVE` — a saved game in `user/openttd/save/` of the artifacts to go on with, such as
  `ailoy-final.sav`, rather than a new game.
* `OPENTTD_SEED`, `OPENTTD_YEAR`, `OPENTTD_MAP_X`, `OPENTTD_MAP_Y` — the new game's random
  seed, its first year (`1950`), and its size as powers of two (`8`, 256 tiles).
* `OPENTTD_SIZE` — the viewer's display, `1280x720` by default.
* `OPENTTD_VNC_PASSWORD` — the viewer's password, `openttd` by default.

Read from `.env` as well.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What the Rust, Python and Node sides share: the skill and the game's side.
SHARED = HERE.parent / "shared"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/gameplay/python/main.py` from the top of the checkout, run again in this one.
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
from virtx import (  # noqa: E402
    Client,
    ImageJobEnd,
    PortForwarder,
    Recipe,
    Sandbox,
    SandboxJobEnd,
    SandboxVolume,
    Stderr,
    Stdout,
    VirtxError,
)
from dotenv import load_dotenv  # noqa: E402

# The image the sandbox boots, built once on the virtx server and kept under this tag.
IMAGE = "ailoy-gameplay:1"

# The VNC server's port in the sandbox.
VNC_PORT = 5900

# The goal when none is given.
GOAL = (
    "Make Ailoy Transport as valuable as you can: build routes that earn, "
    "grow the ones that work, and have the loan paid back by the end if you can."
)

# Rounds at most, so that an agent that answers at once each time does not go on forever.
MAX_ROUNDS = 200

# The settings `start.sh` reads, passed on to it when they are set here.
GAME_SETTINGS = [
    "OPENTTD_SEED",
    "OPENTTD_YEAR",
    "OPENTTD_MAP_X",
    "OPENTTD_MAP_Y",
    "OPENTTD_ADMIN_PASSWORD",
]

INSTRUCTION = (
    "You play OpenTTD: a game is running in your sandbox, and one company in it is yours. "
    "Read the openttd skill before anything else, and play with its `ttd.py`.\n\n"
    "You play in rounds. In each, look at how the company is doing, decide what to build or "
    "change, do it, and let time pass with `wait`, a month or two at a time, checking the "
    "report each time. End a round with a few lines on what you did and how it is going; the "
    "next begins with the date.\n\n"
    "# Artifacts\n\n"
    "Path: `/artifacts`\n\n"
    "Keep your notes in `/artifacts/notes.md`: the routes, their stations, depots and "
    "vehicles, what each earns, and your plan. Update them as you go. Command output from "
    "earlier rounds drops out of what you remember; the notes do not."
)


async def main(goal: str) -> None:
    years = int(os.environ.get("OPENTTD_YEARS", "3"))
    size = os.environ.get("OPENTTD_SIZE", "1280x720")
    password = os.environ.get("OPENTTD_VNC_PASSWORD", "openttd")
    save = os.environ.get("OPENTTD_SAVE", "")

    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()

    # The image, built once: the game, and the free graphics, sounds and music it needs, and
    # Python for `ttd.py`. The viewer's screen and its screenshots are the `display` abin's.
    # The server answers a tag in its full form, as `docker.io/library/ailoy-gameplay:1`.
    images = await client.list_image()
    if not any(image.reference.split("/")[-1] == IMAGE for image in images):
        print(f"Building {IMAGE}, once; it takes a few minutes.")
        recipe = Recipe("debian:trixie-slim").run(
            "apt-get update "
            "&& apt-get install -y --no-install-recommends "
            "openttd openttd-opengfx openttd-opensfx openttd-openmsx python3 "
            "&& rm -rf /var/lib/apt/lists/*"
        )
        async for entry in client.build_image(recipe, IMAGE):
            if isinstance(entry.entry, ImageJobEnd) and entry.entry.error:
                raise RuntimeError(f"building {IMAGE}: {entry.entry.error}")

    # The volumes, made the first time and kept on the server: `409` says one is there already.
    volumes = []
    for name in ["ailoy-gameplay", "ailoy-gameplay-artifacts"]:
        try:
            volumes.append(await client.create_volume(name))
        except VirtxError as e:
            if e.status != 409:
                raise
            volumes.append(await client.get_volume(name))
    files, artifacts = volumes

    # `shared/`, every file of it, onto the first.
    for path in sorted(SHARED.rglob("*")):
        to = path.relative_to(SHARED).as_posix()
        if path.is_dir():
            await files.create_dir_all(to)
        else:
            await files.write_file(to, path.read_bytes())

    sandbox = await client.create_sandbox(
        IMAGE,
        volumes=[
            SandboxVolume(volume=files.name, path="/skills/openttd", readonly=True),
            SandboxVolume(volume=artifacts.name, path="/artifacts"),
        ],
        ports=[VNC_PORT],
        abins=["display"],
        vcpus=2,
        memory_mib=2048,
    )

    # An exec takes no environment of its own, so the settings go through `env`.
    start = ["env"]
    start += [
        f"{name}={os.environ[name]}" for name in GAME_SETTINGS if name in os.environ
    ]
    start += ["sh", "/skills/openttd/server/start.sh", size, password, save]
    # The game is one job, running for as long as the game does; `start.sh` says `ready` once
    # it is up. Leaving its log leaves the job running.
    async for entry in sandbox.exec(start):
        match entry.entry:
            case Stdout(line="ready"):
                break
            case Stdout(line=line) | Stderr(line=line):
                print(line)
            case SandboxJobEnd(code=code):
                raise RuntimeError(f"OpenTTD did not start (exit {code})")
    viewer = PortForwarder.open(client, sandbox.ports[0].url)
    print(
        f"OpenTTD is running. Watch it at vnc://{viewer.address} (password: {password})."
    )

    date = await game_date(sandbox)
    until = year_of(date) + years

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5"))
        .instruction(INSTRUCTION)
        .sandbox(sandbox)
        .skill("/skills/openttd")
        # Many commands a round, each with its output: keep the last two rounds' results
        # whole, and of the screenshots the agent reads, the latest ten once more than
        # thirty have built up.
        .context_manager(max_input_tokens=80_000, preserve_recent_turns=2)
        .build()
    )

    prompt = (
        f"It is {date}, and the game is yours until {until}-01-01. {goal}\n\n"
        "Start by reading the skill, then look at the company, the largest towns and "
        "the industries, and build a first route that will earn."
    )
    for round in range(1, MAX_ROUNDS + 1):
        print(f"\n=== Round {round}: {date} ===\n")
        await play(agent, prompt)

        date = await game_date(sandbox)
        if year_of(date) >= until:
            print(f"\n=== The game has reached {date} ===")
            break
        prompt = (
            f"It is {date}; the game is yours until {until}-01-01. Go on: read your "
            "notes, check the report, then fix what needs fixing, grow what earns, and "
            "let time pass."
        )

    async for entry in sandbox.exec(
        ["python3", "/skills/openttd/ttd.py", "save", "ailoy-final"], timeout_ms=60_000
    ):
        if isinstance(entry.entry, Stdout):
            print(entry.entry.line)

    # The artifacts, every file of them, to `artifacts/` here.
    dirs = [""]
    while dirs:
        at = dirs.pop()
        (PROJECT / "artifacts" / at).mkdir(parents=True, exist_ok=True)
        for entry in await artifacts.read_dir(at):
            if entry.metadata.is_dir:
                dirs.append(f"{at}{entry.name}/")
            else:
                data = await artifacts.read_file(f"{at}{entry.name}")
                (PROJECT / "artifacts" / at / entry.name).write_bytes(data)
    print("The game is still there to look at. Press Enter to stop.")
    await asyncio.to_thread(sys.stdin.readline)


async def play(agent: ailoy.Agent, prompt: str) -> None:
    """Run one round of the agent, printing what it says and does."""
    async for output in agent.run(prompt):
        # The run ends on any other reason as well, and without this it ends in silence.
        if output["finish_reason"]["type"] == "length":
            print("(the reply was cut off at the token limit)", file=sys.stderr)
        message = output["message"]
        if message["role"] == "assistant":
            for part in message["contents"]:
                if part["type"] == "text":
                    print(part["text"])
            for call in message.get("tool_calls") or []:
                function = call["function"]
                print(f"→ {function['name']} {json.dumps(function['arguments'])}")
        # A screenshot is an image part: its bytes are no use on a terminal. And the rest
        # only as much as shows what came back.
        elif message["role"] == "tool":
            for part in message["contents"]:
                if part["type"] == "image":
                    print("← [image]")
                else:
                    text = json.dumps(part, ensure_ascii=False)
                    more = " …" if len(text) > 600 else ""
                    print(f"← {text[:600]}{more}")
        sys.stdout.flush()


async def game_date(sandbox: Sandbox) -> str:
    """The game's date, as `YYYY-MM-DD`."""
    stdout = ""
    async for entry in sandbox.exec(
        ["python3", "/skills/openttd/ttd.py", "--json", "status"], timeout_ms=60_000
    ):
        match entry.entry:
            case Stdout(line=line):
                stdout += line
            case Stderr(line=line):
                print(line, file=sys.stderr)
            case SandboxJobEnd(code=code) if code != 0:
                raise RuntimeError("asking the game its date failed")
    return json.loads(stdout)["date"]


def year_of(date: str) -> int:
    return int(date.split("-")[0])


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:]) or GOAL))
