"""Design 3D parts with an agent that writes CadQuery, looks at what it built and fixes it, in
a console.

    uv run main.py
    uv run main.py "A wall mount for a 60 mm fan, with four M3 screw holes"

The Python side of the Rust `cad` example in `examples/rust/cad`, whose `main.rs` has the long
form. The skill is `SKILL.md` and `render.py`, the ones the three sides share in
`examples/shared/cad`, mounted at `/skills/cad` from memory: the agent writes the model script,
`render.py` checks it and draws it from four sides, and the agent reads the pictures with its
`imgread` tool. There is no model to download and no GPU needed. What a run uses is in this
folder, which is a uv project of its own (`pyproject.toml`) that `uv run` sets up with ailoy
from this checkout; run it from here.

* `context/` at `/context`, read-only — what the request is about, when it is not in the
  prompt: a sketch, a photo of the thing it has to fit, the STEP of a part to mate with.
* `artifacts/` at `/artifacts`, writable — the script, the STEP, STL and GLB files, and the
  pictures.

Environment:

* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its
  provider's API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...). It has to
  take images, or it cannot see what it built.

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
SHARED = HERE.parents[1] / "shared" / "cad"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/python/cad/main.py` from the top of the checkout, run again in this one.
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
from ailoy.virtx import ConsoleClient, Directory, HostMount, Recipe  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

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
    # `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for name in ["context", "artifacts", "skill"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)

    console = await (
        ConsoleClient.builder()
        .image(
            Recipe("python:3.12-slim-trixie")
            # OpenCascade's wheel links libGL and libX11, which the slim image leaves out.
            .step(
                "apt-get update && apt-get install -y --no-install-recommends "
                "libgl1 libx11-6 && rm -rf /var/lib/apt/lists/*"
            )
            .step("pip install --no-cache-dir vtk==9.6.2")
            .step("pip install --no-cache-dir cadquery-ocp==7.9.3.1.1")
            .step(
                "pip install --no-cache-dir --no-deps cadquery==2.8.0 "
                "&& pip install --no-cache-dir casadi ezdxf multimethod nlopt pyparsing "
                "runtype scipy typing_extensions trimesh numpy pillow"
            )
        )
        .mount_readonly(
            HostMount(
                Directory()
                .with_file("SKILL.md", (SHARED / "SKILL.md").read_bytes())
                .with_file("render.py", (SHARED / "render.py").read_bytes()),
                PROJECT / "skill",
            ),
            "/skills/cad",
        )
        .mount_readonly(PROJECT / "context", "/context")
        .mount(PROJECT / "artifacts", "/artifacts")
        .vcpus(4)
        .memory_mib(4096)
        .build()
    )

    agent = await (
        ailoy.AgentBuilder(
            os.environ.get("AILOY_MODEL", "bedrock/global.openai.gpt-6-astra")
        )
        .instruction(INSTRUCTION)
        # A whole model script is one `write`, and the model thinks before it, which counts
        # against the same limit: far more than the 8192 tokens a reply gets by default.
        .max_tokens(64000)
        .system_tools()
        .console(console)
        .skill("/skills/cad")
        .build()
    )

    async with console:
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


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:]) or QUERY))
