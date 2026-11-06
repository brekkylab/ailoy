"""Segment images and videos with SAM3 through ncnn on the guest's Vulkan device, as an
agent's skill.

    uv run main.py "Find every cat in the photos in context and mask them"

The Python side of the Rust `sam3` example in `examples/sam3/rust`, whose `main.rs` has the
long form. The skill is `SKILL.md` and `run_sam3.py`, which the agent runs with its `shell`
tool, put each run onto the volume `ailoy-sam3-skill` of the virtx server mounted at
`/skills/sam3`. They and `prepare_model.py`, which downloads and converts the model into `data/`
on the first run, are the ones the three sides share in `examples/sam3/shared`. What a run uses
is in this folder, which is a uv project of its own (`pyproject.toml`) that `uv run` sets up
with ailoy from this checkout; run it from here.

* The volume `ailoy-sam3-models` at `/models`, read-only — `data/ncnn/` here, the converted
  models, put there each run.
* The volume `ailoy-sam3-context` at `/context`, read-only — `context/` here, put there each
  run: the images and frames to segment, when they are not in the prompt.
* The volume `ailoy-sam3-artifacts` at `/artifacts`, writable — where what the agent hands
  back goes. It stays on the server between runs, and is copied to `artifacts/` here at the
  end.

Environment:

* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to be
  set up as `docs/src/content/docs/guide/model-providers.mdx` says.

Read from `.env` as well.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
SHARED = HERE.parent / "shared"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/sam3/python/main.py` from the top of the checkout, run again in this one.
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
IMAGE = "ailoy-sam3:1"

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


def _bytes(value: object) -> str:
    """What `json.dumps` prints for an image's bytes, as `imgread` hands them back."""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


async def prepare(shared: Path, project: Path) -> None:
    """Download and convert the models into `project/data`."""
    # In this interpreter: the project's environment has what it needs too.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        shared / "prepare_model.py",
        project / "data",
        cwd=shared,
    )
    if await proc.wait() != 0:
        sys.exit(f"preparing the models: exit status {proc.returncode}")


async def main(prompt: str) -> None:
    await prepare(SHARED, PROJECT)
    (PROJECT / "context").mkdir(parents=True, exist_ok=True)

    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()

    # The image, built once. The server answers a tag in its full form, as
    # `docker.io/library/ailoy-sam3:1`.
    images = await client.list_image()
    if not any(image.reference.split("/")[-1] == IMAGE for image in images):
        print(f"Building {IMAGE}, once; it takes a few minutes.", flush=True)
        recipe = (
            Recipe("python:3.12-slim-trixie")
            # Mesa from backports: venus passes VK_KHR_shader_bfloat16 and VK_KHR_cooperative_matrix
            # through from 26.0 on, and trixie itself has 25.0.
            .run(
                "echo 'deb http://deb.debian.org/debian trixie-backports main' "
                "> /etc/apt/sources.list.d/backports.list "
                "&& apt-get update && apt-get install -y --no-install-recommends "
                "-t trixie-backports mesa-vulkan-drivers "
                "&& apt-get install -y --no-install-recommends libvulkan1 "
                "&& rm -rf /var/lib/apt/lists/*"
            )
            # Headless OpenCV: `opencv-python` needs libGL, which the slim image lacks. `ncnn` asks for
            # it by name, and both wheels install the same `cv2` package, so `--no-deps` on `ncnn` and
            # its own dependencies spelled out, opencv aside.
            .run(
                "pip install --no-cache-dir av numpy opencv-python-headless pillow "
                "portalocker requests tokenizers tqdm "
                "&& pip install --no-cache-dir --no-deps ncnn"
            )
        )
        async for entry in client.build_image(recipe, IMAGE):
            if isinstance(entry.entry, ImageJobEnd) and entry.entry.error:
                raise RuntimeError(f"building {IMAGE}: {entry.entry.error}")

    # The volumes, made the first time and kept on the server: `409` says one is there already.
    volumes = []
    for name in [
        "ailoy-sam3-skill",
        "ailoy-sam3-models",
        "ailoy-sam3-context",
        "ailoy-sam3-artifacts",
    ]:
        try:
            volumes.append(await client.create_volume(name))
        except VirtxError as e:
            if e.status != 409:
                raise
            volumes.append(await client.get_volume(name))
    skill, models, context, artifacts = volumes

    # The skill onto the first.
    for name in ["SKILL.md", "run_sam3.py"]:
        await skill.write_file(name, (SHARED / name).read_bytes())

    # `data/ncnn/`, every file of it, onto the second.
    for path in sorted((PROJECT / "data" / "ncnn").rglob("*")):
        to = path.relative_to(PROJECT / "data" / "ncnn").as_posix()
        if path.is_dir():
            await models.create_dir_all(to)
        else:
            await models.write_file(to, path.read_bytes())

    # `context/`, every file of it, onto the third.
    for path in sorted((PROJECT / "context").rglob("*")):
        to = path.relative_to(PROJECT / "context").as_posix()
        if path.is_dir():
            await context.create_dir_all(to)
        else:
            await context.write_file(to, path.read_bytes())

    sandbox = await client.create_sandbox(
        IMAGE,
        volumes=[
            SandboxVolume(volume=models.name, path="/models", readonly=True),
            SandboxVolume(volume=skill.name, path="/skills/sam3", readonly=True),
            SandboxVolume(volume=context.name, path="/context", readonly=True),
            SandboxVolume(volume=artifacts.name, path="/artifacts"),
        ],
        network=True,
        gpu=True,
        vcpus=2,
        memory_mib=4096,
        gpu_memory_mib=12288,
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "openai/gpt-6-astra"))
        .instruction(INSTRUCTION)
        .web_fetch_tool()
        .web_search_tool([])
        .sandbox(sandbox)
        .skill("/skills/sam3")
        .build()
    )

    async for output in agent.run(prompt):
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
        # A run prints a summary of what it found and where the masks went, and ncnn's
        # device log: shown whole.
        elif message["role"] == "tool":
            for part in message["contents"]:
                print(
                    f"← {json.dumps(part, indent=2, ensure_ascii=False, default=_bytes)}"
                )
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
    asyncio.run(main(" ".join(sys.argv[1:])))
