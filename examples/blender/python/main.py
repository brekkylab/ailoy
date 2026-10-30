"""Build 3D scenes in a live Blender that you watch over VNC, with an agent that drives it,
renders what it built, looks at it and fixes it.

    uv run main.py
    uv run main.py "A low-poly fox sitting on a mossy rock, for a game"

The Python side of the Rust `blender` example in `examples/blender/rust`, whose `main.rs` has
the long form. Blender runs on a VNC display in a console of its own, published here at
`localhost:5901` (password `ailoy`): open `vnc://localhost:5901` to watch. The agent works in a
second console and sends its scripts to Blender with `run.py`, at port 8765. The skill is
`SKILL.md`, `server.py`, `run.py` and `start.sh`, the ones the three sides share in
`examples/blender/shared`, mounted at `/skills/blender` from memory in both consoles. What a
run uses is in this folder, which is a uv project of its own (`pyproject.toml`) that `uv run`
sets up with ailoy from this checkout; run it from here.

* `context/` at `/context`, read-only — what the request is about, when it is not in the
  prompt: a reference picture, a concept sketch, a `.blend` to start from.
* `artifacts/` at `/artifacts`, writable — the scripts, `scene.blend` and `scene.glb`, the
  renders and the turntable.

Environment:

* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to
  be set up as `docs/guide/model-providers.md` says. It has to take images, or it cannot see
  what it built.

Read from `.env` as well.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Absolute, because a mount is named to the server as a `file://` URL.
HERE = Path(__file__).resolve().parent
# What the Rust, Python and Node sides share: the skill.
SHARED = HERE.parent / "shared"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/blender/python/main.py` from the top of the checkout, run again in this one.
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
from virtx import ConsoleClient, Directory, HostMount, Recipe, ensure_virtx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# The request when none is given.
QUERY = (
    "Build a small isometric diorama of a ramen stall at night: a square base about 4 m "
    "across, the stall with a counter, three stools and a noren curtain over the front, paper "
    "lanterns that glow, a steaming bowl on the counter and a vending machine beside it. Warm "
    "light from inside the stall, cool moonlight from above. Frame it from the front "
    "three-quarters, render it, and give me a turntable."
)

INSTRUCTION = (
    "# Context\n\n"
    "Path: `/context`\n\n"
    "This folder holds the data and context the user wants to share with you. "
    "When the user refers to something whose context you cannot figure out, the files in this folder might help. "
    "The information that settles the answer may be here too, and so may hints toward it, so look through this folder for them.\n\n"
    "# Artifacts\n\n"
    "Path: `/artifacts`\n\n"
    "This folder is where what the user asked for goes. "
    "Write every result here, such as a report, a figure, or the file the user came for. "
    "Everything in this folder is collected and handed back to the user, and a result left anywhere else is not delivered."
)

SKILL_FILES = ["SKILL.md", "server.py", "run.py", "start.sh"]


async def main(prompt: str) -> None:
    # `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for name in ["context", "artifacts", "skill"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)

    files = Directory()
    for name in SKILL_FILES:
        files = files.with_file(name, (SHARED / name).read_bytes())
    # One mount, shared: both consoles see the same skill.
    skill = HostMount(files, PROJECT / "skill")

    # The console server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none.
    await ensure_virtx()
    # The agent's console is stopped between its tool calls, and Blender with it would be, so
    # Blender has a console the agent does not hold.
    blender = await (
        ConsoleClient.builder()
        .image(
            Recipe("debian:sid-slim")
            # Blender 5.2 is in sid for every architecture. `mesa-vulkan-drivers` carries the
            # guest's venus ICD and llvmpipe, `libvulkan1` the loader; Xvnc is the display.
            .step(
                "apt-get update && apt-get install -y --no-install-recommends "
                "blender python3-numpy mesa-vulkan-drivers libvulkan1 "
                "tigervnc-standalone-server tigervnc-tools && rm -rf /var/lib/apt/lists/*"
            )
            # Saved preferences skip the first-run splash, and renders open no window.
            .step(
                'blender -b --factory-startup --python-expr "import bpy; '
                "v = bpy.context.preferences.view; v.show_splash = False; "
                "v.render_display_type = 'NONE'; bpy.ops.wm.save_userpref()\""
            )
        )
        .mount_readonly(skill, "/skills/blender")
        .mount_readonly(PROJECT / "context", "/context")
        .mount(PROJECT / "artifacts", "/artifacts")
        # What comes in is the viewer, at 5901 here to leave 5900 to macOS Screen Sharing, and
        # `run.py` from the agent's console.
        .network(True)
        .ports(["5901:5900", "8765:8765"])
        .gpu(True)
        .vcpus(4)
        .memory_mib(8192)
        .gpu_memory_mib(8192)
        .build()
    )

    async with blender:
        # Blender up before the agent's first turn, so the person can watch from the start.
        started = await blender.exec(["sh", "/skills/blender/start.sh"], 180_000)
        if started.code != 0:
            sys.exit(f"starting Blender: {started.stderr.decode(errors='replace')}")
        print("Blender is up: open vnc://localhost:5901 (password: ailoy) to watch.\n")

        # Python for `run.py`, and a network to reach the port Blender's console published.
        console = await (
            ConsoleClient.builder()
            .image(Recipe("python:3.12-slim-trixie"))
            .mount_readonly(skill, "/skills/blender")
            .mount_readonly(PROJECT / "context", "/context")
            .mount(PROJECT / "artifacts", "/artifacts")
            .network(True)
            .build()
        )
        async with console:
            agent = await (
                ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "openai/gpt-6-astra"))
                .instruction(INSTRUCTION)
                .system_tools()
                .console(console)
                .skill("/skills/blender")
                .build()
            )

            async for output in agent.run(prompt):
                # A token-limit cutoff also ends the run; without this it ends silently.
                if output["finish_reason"]["type"] == "length":
                    print("(the reply was cut off at the token limit)", file=sys.stderr)
                message = output["message"]
                if message["role"] == "assistant":
                    for part in message["contents"]:
                        if part["type"] == "text":
                            print(part["text"])
                    for call in message.get("tool_calls") or []:
                        function = call["function"]
                        print(
                            f"→ {function['name']} {json.dumps(function['arguments'], indent=2)}"
                        )
                # A `read` of a picture is an image part: its bytes are no use on a terminal.
                elif message["role"] == "tool":
                    for part in message["contents"]:
                        if part["type"] == "image":
                            print("← [image]")
                        else:
                            print(f"← {json.dumps(part, indent=2, ensure_ascii=False)}")
                sys.stdout.flush()

        # Blender's console, and Blender in it, goes when this block ends.
        await asyncio.to_thread(
            input,
            "\nBlender stays open at vnc://localhost:5901. Press Enter to close it.",
        )


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:]) or QUERY))
