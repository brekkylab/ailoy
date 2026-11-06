"""Answer typed questions with Laya through ncnn on the guest's Vulkan device, as an agent's
skill.

    uv run main.py
    uv run main.py "Route the tickets in /context/inbox, but send anything about contracts to legal"

The Python side of the Rust `laya` example in `examples/laya/rust`, whose `main.rs` has the
long form. The skill is `SKILL.md` and `run_laya.py`, put each run onto the volume
`ailoy-laya-skill` of the virtx server mounted at `/skills/laya`, which the agent runs with its
`shell` tool. They and `prepare_model.py`, which downloads and converts the model into `data/` on
the first run, are the ones the three sides share in `examples/laya/shared`. `data/ncnn/` goes
each run onto the volume `ailoy-laya-models` at `/models`, read-only. What a run uses is in this folder, which is a uv project of its own
(`pyproject.toml`) that `uv run` sets up with ailoy from this checkout; run it from here.

Without a prompt the agent routes a support inbox: it asks Laya which department each ticket in
`shared/tickets/` belongs to and files a copy in that department's folder.

* The volume `ailoy-laya-context` at `/context`, read-only — `context/`, what to decide on, when
  it is not in the prompt, put there each run. `context/inbox/` is a fresh copy of
  `shared/tickets/` on every run.
* The volume `ailoy-laya-artifacts` at `/artifacts`, writable — where what the agent hands back
  goes: the tickets filed under `routes/<department>/`, emptied at the start of every run. It
  stays on the server between runs, and is copied to `artifacts/` here at the end.

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
# `python examples/laya/python/main.py` from the top of the checkout, run again in this one.
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
IMAGE = "ailoy-laya:1"

# The request when none is given.
QUERY = (
    "Route every ticket in /context/inbox to the department that should handle it, by what the "
    "customer needs rather than the words they use: engineering, finance, sales, legal or "
    "marketing. Decide each one with Laya, copy the ticket into /artifacts/routes/<department>/, "
    "and write what went where to /artifacts/routing.md."
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
    """Download and convert the model into `project/data`."""
    # In this interpreter: the project's environment has what it needs too. `transformers`
    # probes for TensorFlow at import, which can hang model construction.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        shared / "prepare_model.py",
        project / "data",
        cwd=shared,
        env={**os.environ, "USE_TF": "0"},
    )
    if await proc.wait() != 0:
        sys.exit(f"preparing the model: exit status {proc.returncode}")


def fill_inbox(shared: Path, project: Path) -> None:
    """Reset `project/context/inbox` to a copy of `shared/tickets`, so every run routes the same
    inbox."""
    inbox = project / "context" / "inbox"
    shutil.rmtree(inbox, ignore_errors=True)
    inbox.mkdir(parents=True)
    for ticket in (shared / "tickets").iterdir():
        if ticket.is_file():
            shutil.copy(ticket, inbox / ticket.name)


def clear_routes(project: Path) -> None:
    """Empty `project/artifacts/routes`, so what is filed there is only this run's."""
    shutil.rmtree(project / "artifacts" / "routes", ignore_errors=True)


async def main(prompt: str) -> None:
    await prepare(SHARED, PROJECT)
    for name in ["context", "artifacts"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)
    fill_inbox(SHARED, PROJECT)
    clear_routes(PROJECT)

    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()

    # The image, built once. The server answers a tag in its full form, as
    # `docker.io/library/ailoy-laya:1`.
    images = await client.list_image()
    if not any(image.reference.split("/")[-1] == IMAGE for image in images):
        print(f"Building {IMAGE}, once; it takes a few minutes.")
        # Debian, not Alpine: PyPI's musllinux ncnn wheel crashes freeing its first `Mat`,
        # where the manylinux (glibc) one runs laya.
        recipe = (
            Recipe("python:3.12-slim-trixie")
            # `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the
            # wheel opens. Mesa from backports: venus passes VK_KHR_shader_bfloat16 through
            # from 26.0 on, and trixie itself has 25.0.
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
        "ailoy-laya-models",
        "ailoy-laya-skill",
        "ailoy-laya-context",
        "ailoy-laya-artifacts",
    ]:
        try:
            volumes.append(await client.create_volume(name))
        except VirtxError as e:
            if e.status != 409:
                raise
            volumes.append(await client.get_volume(name))
    models, skill, context, artifacts = volumes

    # The skill, `SKILL.md` and `run_laya.py` of `shared/`, onto its own.
    for name in ["SKILL.md", "run_laya.py"]:
        await skill.write_file(name, (SHARED / name).read_bytes())
    # A fresh inbox and no routes, as on this side.
    await context.remove_file("inbox", missing_ok=True)
    await artifacts.remove_file("routes", missing_ok=True)
    # `data/ncnn/` and `context/`, every file of them, onto theirs.
    for volume, root in [
        (models, PROJECT / "data" / "ncnn"),
        (context, PROJECT / "context"),
    ]:
        for path in sorted(root.rglob("*")):
            to = path.relative_to(root).as_posix()
            if path.is_dir():
                await volume.create_dir_all(to)
            else:
                await volume.write_file(to, path.read_bytes())

    sandbox = await client.create_sandbox(
        IMAGE,
        volumes=[
            SandboxVolume(volume=models.name, path="/models", readonly=True),
            SandboxVolume(volume=skill.name, path="/skills/laya", readonly=True),
            SandboxVolume(volume=context.name, path="/context", readonly=True),
            SandboxVolume(volume=artifacts.name, path="/artifacts"),
        ],
        vcpus=2,
        memory_mib=4096,
        gpu=True,
        gpu_memory_mib=12288,
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5"))
        .instruction(INSTRUCTION)
        .system_tools()
        .web_fetch_tool()
        .web_search_tool([])
        .sandbox(sandbox)
        .skill("/skills/laya")
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
        # Laya's answers are small, and they are what the reply is made from: shown whole.
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
    asyncio.run(main(" ".join(sys.argv[1:]) or QUERY))
