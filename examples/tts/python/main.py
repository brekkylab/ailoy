"""Speak a text in a voice described in words with Qwen3-TTS, through ncnn on the guest's Vulkan
device, as an agent's skill.

    uv run main.py

The Python side of the Rust `tts` example in `examples/tts/rust`, whose `main.rs` has the
long form. There is no prompt: the agent reads `text.txt` and `instruct.txt` from the context
folder, speaks the text in that voice and hands back the WAV. The skill is `SKILL.md` and
`run_tts.py`, put each run onto a volume of the virtx server, `ailoy-tts-skill`, mounted at
`/skills/tts`. They, `prepare_model.py`, which downloads and converts the models into `data/`
on the first run, and `context_example/` are the ones the three sides share in
`examples/tts/shared`. What a run uses is in this folder, which is a uv project of its own
(`pyproject.toml`) that `uv run` sets up with ailoy from this checkout; run it from here.

* The volume `ailoy-tts-models` at `/models`, read-only — `data/ncnn/`, put there each run.
* The volume `ailoy-tts-context` at `/context`, read-only — `context/`, the text and the
  instruction, put there each run; `context_example/` is copied into `context/` when it is
  empty.
* The volume `ailoy-tts-artifacts` at `/artifacts`, writable — where what the agent hands back
  goes. It stays on the server between runs, and is copied to `artifacts/` here at the end.

Qwen3-TTS and its weights are the Qwen team's, under Apache-2.0.

Environment:

* `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend has
  to be set up as `docs/src/content/docs/guide/model-providers.mdx` says.

Read from `.env` as well.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
SHARED = HERE.parent / "shared"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/tts/python/main.py` from the top of the checkout, run again in this one.
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
from virtx import Client, ImageJobEnd, Recipe, SandboxVolume, VirtxError  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# The image the sandbox boots, built once on the virtx server and kept under this tag.
IMAGE = "ailoy-tts:1"

QUERY = (
    "Speak the text in /context in the voice and manner its instruction describes, "
    "and hand back the speech as a WAV file."
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


async def main() -> None:
    await prepare(SHARED, PROJECT)
    for name in ["context", "artifacts"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)
    # Something to say, for a first run.
    context = PROJECT / "context"
    if not any(context.iterdir()):
        for entry in (SHARED / "context_example").iterdir():
            shutil.copy(entry, context / entry.name)

    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()

    # The image, built once. Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
    # The server answers a tag in its full form, as `docker.io/library/ailoy-tts:1`.
    images = await client.list_image()
    if not any(image.reference.split("/")[-1] == IMAGE for image in images):
        print(f"Building {IMAGE}, once; it takes a few minutes.")
        recipe = (
            Recipe("python:3.12-slim-trixie")
            # `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the
            # wheel opens. Mesa from backports: 26.0 against trixie's 25.0.
            .run(
                "echo 'deb http://deb.debian.org/debian trixie-backports main' "
                "> /etc/apt/sources.list.d/backports.list "
                "&& apt-get update && apt-get install -y --no-install-recommends "
                "-t trixie-backports mesa-vulkan-drivers "
                "&& apt-get install -y --no-install-recommends libvulkan1 "
                "&& rm -rf /var/lib/apt/lists/*"
            )
            .run("pip install --no-cache-dir ncnn numpy tokenizers")
        )
        async for entry in client.build_image(recipe, IMAGE):
            if isinstance(entry.entry, ImageJobEnd) and entry.entry.error:
                raise RuntimeError(f"building {IMAGE}: {entry.entry.error}")

    # The volumes, made the first time and kept on the server: `409` says one is there already.
    volumes = []
    for name in [
        "ailoy-tts-models",
        "ailoy-tts-skill",
        "ailoy-tts-context",
        "ailoy-tts-artifacts",
    ]:
        try:
            volumes.append(await client.create_volume(name))
        except VirtxError as e:
            if e.status != 409:
                raise
            volumes.append(await client.get_volume(name))
    models, skill, context_volume, artifacts = volumes

    # `data/ncnn/` and `context/`, every file of them, onto theirs.
    for source, volume in [
        (PROJECT / "data" / "ncnn", models),
        (context, context_volume),
    ]:
        for path in sorted(source.rglob("*")):
            to = path.relative_to(source).as_posix()
            if path.is_dir():
                await volume.create_dir_all(to)
            else:
                await volume.write_file(to, path.read_bytes())
    # The skill, from `shared/`.
    for name in ["SKILL.md", "run_tts.py"]:
        await skill.write_file(name, (SHARED / name).read_bytes())

    sandbox = await client.create_sandbox(
        IMAGE,
        volumes=[
            SandboxVolume(volume=models.name, path="/models", readonly=True),
            SandboxVolume(volume=skill.name, path="/skills/tts", readonly=True),
            SandboxVolume(volume=context_volume.name, path="/context", readonly=True),
            SandboxVolume(volume=artifacts.name, path="/artifacts"),
        ],
        gpu=True,
        vcpus=2,
        # The talker is 2.8 GB on the device in fp16, and its cache and the codec's buffers
        # come on top of it there; the run itself takes under 1 GB of memory.
        memory_mib=4096,
        gpu_memory_mib=8192,
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5"))
        .instruction(INSTRUCTION)
        .sandbox(sandbox)
        .skill("/skills/tts")
        .build()
    )

    async for output in agent.run(QUERY):
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
        # A run prints where the speech went and how long it is, and ncnn's device log
        # and its progress: shown whole.
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
    asyncio.run(main())
