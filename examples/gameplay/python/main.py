"""An agent that plays OpenTTD, running in a console, while you watch it over VNC.

    uv run main.py
    uv run main.py "Connect every town over 1000 people by air"

The Python side of the Rust `gameplay` example in `examples/gameplay/rust`, whose `main.rs` has
the long form. The game runs as a dedicated server in a console of its own, and the agent runs
a company in it from another, through the skill's `ttd.py`. The skill (`SKILL.md`, `ttd.py`,
`admin.py`) and the game's side (`server/`: `start.sh`, the Game Script, the AI and `shot.py`)
are the ones the three sides share in `examples/gameplay/shared`; the skill is mounted from
memory at `/skills/openttd` in both consoles. What a run uses is in this folder, which is a uv
project of its own (`pyproject.toml`) that `uv run` sets up with ailoy from this checkout; run
it from here.

* `artifacts/` at `/artifacts` in both, writable — the agent's notes, the screenshots it took,
  the saved games in `user/openttd/save/` (the last as `ailoy-final.sav`), and the logs.

Watch at `vnc://localhost:5901` (Screen Sharing on macOS, or any VNC viewer) with the password
`openttd`.

Environment:

* `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
  API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...). It reads screenshots
  when it looks, so it should take images.
* `OPENTTD_YEARS` — how many years of the game the agent plays, `3` by default. A game month
  takes a minute to pass.
* `OPENTTD_SAVE` — a saved game in `artifacts/user/openttd/save/` to go on with, such as
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

# Absolute, because a mount is named to the server as a `file://` URL.
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
from ailoy.virtx import ConsoleClient, Directory, HostMount, Recipe, ensure_virtx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# The port a VNC viewer connects to here, and the VNC server's in the game's console.
VIEWER_PORT = 5901
VNC_PORT = 5900

# The game's admin port and `shot.py`'s, published at the same numbers here so that
# `admin.py` names one address for both consoles.
ADMIN_PORT = 3977
SHOT_PORT = 5902

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
    "You play OpenTTD: a game is running in the console, and one company in it is yours. "
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

    # `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for name in ["artifacts", "skill"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)

    # The console server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none.
    await ensure_virtx()
    # One mount, shared: both consoles see the same skill.
    skill = HostMount(
        Directory()
        .with_file("SKILL.md", (SHARED / "SKILL.md").read_bytes())
        .with_file("ttd.py", (SHARED / "ttd.py").read_bytes())
        .with_file("admin.py", (SHARED / "admin.py").read_bytes()),
        PROJECT / "skill",
    )

    game = await (
        ConsoleClient.builder()
        .image(
            # All of it in `main`: the game, and the free graphics, sounds and music it needs,
            # a display for the viewer, and ImageMagick for the agent's screenshots of it.
            Recipe("debian:trixie-slim").step(
                "apt-get update "
                "&& apt-get install -y --no-install-recommends "
                "openttd openttd-opengfx openttd-opensfx openttd-openmsx xvfb x11vnc python3 "
                "imagemagick "
                "&& rm -rf /var/lib/apt/lists/*"
            )
        )
        .mount_readonly(SHARED / "server", "/example")
        .mount_readonly(skill, "/skills/openttd")
        .mount(PROJECT / "artifacts", "/artifacts")
        # The build's `apt-get` runs with the session's network. What comes in is the viewer,
        # and the agent's console at the admin port and `shot.py`.
        .ports(
            [
                f"{VIEWER_PORT}:{VNC_PORT}",
                f"{ADMIN_PORT}:{ADMIN_PORT}",
                f"{SHOT_PORT}:{SHOT_PORT}",
            ]
        )
        .vcpus(2)
        .memory_mib(2048)
        .build()
    )

    async with game:
        # An exec takes no environment of its own, so the settings go through `env`.
        start = ["env"]
        start += [f"{name}={os.environ[name]}" for name in GAME_SETTINGS if name in os.environ]
        start += ["sh", "/example/start.sh", size, password, save]
        out = await game.exec(start, 180_000)
        print(out.stdout.decode(errors="replace"), end="")
        if out.code != 0:
            raise RuntimeError(f"starting OpenTTD:\n{out.stderr.decode(errors='replace')}")
        print(
            f"OpenTTD is running. Watch it at vnc://localhost:{VIEWER_PORT} "
            f"(password: {password})."
        )

        date = await game_date(game)
        until = year_of(date) + years

        # Python for `ttd.py`, and a network to reach the ports the game's console published.
        console = await (
            ConsoleClient.builder()
            .image(Recipe("python:3.12-slim-trixie"))
            .mount_readonly(skill, "/skills/openttd")
            .mount(PROJECT / "artifacts", "/artifacts")
            .network(True)
            .build()
        )

        agent = await (
            ailoy.AgentBuilder(
                os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5")
            )
            .instruction(INSTRUCTION)
            .max_tokens(32000)
            .system_tools()
            .console(console)
            .skill("/skills/openttd")
            # Many commands a round, each with its output: keep the last two rounds whole.
            .context_manager(max_input_tokens=80_000, preserve_recent_turns=2)
            .build()
        )

        async with console:
            prompt = (
                f"It is {date}, and the game is yours until {until}-01-01. {goal}\n\n"
                "Start by reading the skill, then look at the company, the largest towns and "
                "the industries, and build a first route that will earn."
            )
            for round in range(1, MAX_ROUNDS + 1):
                print(f"\n=== Round {round}: {date} ===\n")
                await play(agent, prompt)

                date = await game_date(game)
                if year_of(date) >= until:
                    print(f"\n=== The game has reached {date} ===")
                    break
                prompt = (
                    f"It is {date}; the game is yours until {until}-01-01. Go on: read your "
                    "notes, check the report, then fix what needs fixing, grow what earns, and "
                    "let time pass."
                )

        out = await game.exec(
            ["python3", "/skills/openttd/ttd.py", "save", "ailoy-final"], 60_000
        )
        print(out.stdout.decode(errors="replace"), end="")
        print("The game is still there to look at. Press Enter to stop.")
        await asyncio.to_thread(sys.stdin.readline)

    # Leaving the `async with` blocks tears the consoles down, game and all.


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


async def game_date(console: ConsoleClient) -> str:
    """The game's date, as `YYYY-MM-DD`."""
    out = await console.exec(
        ["python3", "/skills/openttd/ttd.py", "--json", "status"], 60_000
    )
    if out.code != 0:
        raise RuntimeError(
            f"asking the game its date:\n{out.stderr.decode(errors='replace')}"
        )
    return json.loads(out.stdout)["date"]


def year_of(date: str) -> int:
    return int(date.split("-")[0])


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:]) or GOAL))
