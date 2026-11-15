"""Design 3D parts with an agent that writes CadQuery, looks at what it built and fixes it, in
a sandbox.

    uv run main.py
    uv run main.py "A wall mount for a 60 mm fan, with four M3 screw holes"

The Python side of the Rust `cad` example in `examples/cad/rust`, whose `main.rs` has the long
form. The skill is `SKILL.md` and `render.py`, the ones the three sides share in
`examples/cad/shared`, put each run onto the volume `ailoy-cad-skill` of the virtx server
mounted at `/skills/cad`: the agent writes the model script,
`render.py` checks it and draws it from four sides, and the agent reads the pictures with its
`imgread` tool. There is no model to download and no GPU needed. What a run uses is in this
folder, which is a uv project of its own (`pyproject.toml`) that `uv run` sets up with ailoy
from this checkout; run it from here.

* The volume `ailoy-cad-context` at `/context`, read-only — `context/` here, put there each
  run: what the request is about, when it is not in the prompt: a sketch, a photo of the
  thing it has to fit, the STEP of a part to mate with.
* The volume `ailoy-cad-artifacts` at `/artifacts`, writable — the script, the STEP, STL and
  GLB files, and the pictures. It stays on the server between runs, and is copied to
  `artifacts/` here at the end.

Environment:

* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to be
  set up as `docs/content/guides/model-providers.mdx` says. It has to take images,
  or it cannot see what it built.

Read from `.env` as well.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What the Rust, Python and Node sides share: the skill.
SHARED = HERE.parent / "shared"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/cad/python/main.py` from the top of the checkout, run again in this one.
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
    Recipe,
    SandboxVolume,
    VirtxError,
)
from dotenv import load_dotenv  # noqa: E402

# The image the sandbox boots, built once on the virtx server and kept under this tag.
IMAGE = "ailoy-cad:1"

# The request when none is given.
QUERY = (
    "Design a gear bearing that prints in one piece, already assembled: a "
    "sun, five planets and a ring, all with herringbone teeth so the planets cannot slide out, "
    "and enough clearance that it turns when it comes off the bed. About 60 mm across and 15 mm "
    "tall, with a hexagonal hole through the sun for a key. Show it assembled, cut in half and "
    "turning."
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


async def main(prompt: str) -> None:
    (PROJECT / "context").mkdir(parents=True, exist_ok=True)

    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()

    # The image, built once. The server answers a tag in its full form, as
    # `docker.io/library/ailoy-cad:1`.
    images = await client.list_image()
    if not any(image.reference.split("/")[-1] == IMAGE for image in images):
        print(f"Building {IMAGE}, once; it takes a few minutes.")
        recipe = (
            Recipe("python:3.12-slim-trixie")
            # OpenCascade's wheel links libGL and libX11, which the slim image leaves out.
            .run(
                "apt-get update && apt-get install -y --no-install-recommends "
                "libgl1 libx11-6 && rm -rf /var/lib/apt/lists/*"
            )
            .run("pip install --no-cache-dir vtk==9.6.2")
            .run("pip install --no-cache-dir cadquery-ocp==7.9.3.1.1")
            .run(
                "pip install --no-cache-dir --no-deps cadquery==2.8.0 "
                "&& pip install --no-cache-dir casadi ezdxf multimethod nlopt pyparsing "
                "runtype scipy typing_extensions trimesh numpy pillow"
            )
        )
        async for entry in client.build_image(recipe, IMAGE):
            if isinstance(entry.entry, ImageJobEnd) and entry.entry.error:
                raise RuntimeError(f"building {IMAGE}: {entry.entry.error}")

    # The volumes, made the first time and kept on the server: `409` says one is there already.
    volumes = []
    for name in ["ailoy-cad-skill", "ailoy-cad-context", "ailoy-cad-artifacts"]:
        try:
            volumes.append(await client.create_volume(name))
        except VirtxError as e:
            if e.status != 409:
                raise
            volumes.append(await client.get_volume(name))
    skill, context, artifacts = volumes

    # The skill onto the first.
    for name in ["SKILL.md", "render.py"]:
        await skill.write_file(name, (SHARED / name).read_bytes())

    # `context/`, every file of it, onto the second.
    for path in sorted((PROJECT / "context").rglob("*")):
        to = path.relative_to(PROJECT / "context").as_posix()
        if path.is_dir():
            await context.create_dir_all(to)
        else:
            await context.write_file(to, path.read_bytes())

    sandbox = await client.create_sandbox(
        IMAGE,
        volumes=[
            SandboxVolume(volume=skill.name, path="/skills/cad", readonly=True),
            SandboxVolume(volume=context.name, path="/context", readonly=True),
            SandboxVolume(volume=artifacts.name, path="/artifacts"),
        ],
        vcpus=4,
        memory_mib=4096,
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "openai/gpt-6-astra"))
        .instruction(INSTRUCTION)
        .sandbox(sandbox)
        .skill("/skills/cad")
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


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:]) or QUERY))
