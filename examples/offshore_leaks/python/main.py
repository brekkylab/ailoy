"""Analyze ICIJ's Offshore Leaks database with an agent that writes and runs its own SQL and
Python, in a sandbox, against the data it is given as context.

    uv run main.py
    uv run main.py "Which South Korean officers appear in more than one leak?"

The Python side of the Rust `offshore_leaks` example in `examples/offshore_leaks/rust`, whose
`main.rs` has the long form. `prepare_data.py` loads ICIJ's CSVs into one DuckDB file in
`context/` on the first run, which the agent queries in place. The skill is `SKILL.md` and
`oldb.py`, which go each run onto the volume `ailoy-offshore-leaks-skill` of the virtx server,
mounted at `/skills/offshore-leaks`. These three are the ones the three sides share in
`examples/offshore_leaks/shared`. What a run uses is in this folder, which is a uv project of
its own (`pyproject.toml`) that `uv run` sets up with ailoy from this checkout; run it from
here.

* The volume `ailoy-offshore-leaks-context` at `/context`, read-only — `context/` here, every
  file of it, put there each run: `offshore_leaks.duckdb`, and whatever else the request is
  about, such as a list of names to look for.
* The volume `ailoy-offshore-leaks-artifacts` at `/artifacts`, writable — where the reports,
  tables and charts go. It stays on the server between runs, and is copied to `artifacts/`
  here at the end.

The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA.
Being in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.

Environment:

* `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads; ICIJ's latest by default.
* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to be
  set up as `docs/content/guides/model-providers.mdx` says.

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
# `python examples/offshore_leaks/python/main.py` from the top of the checkout, run again in
# this one.
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
IMAGE = "ailoy-offshore-leaks:1"

# The request when none is given.
QUERY = (
    "Which intermediaries set up the most entities in the Panama Papers, and "
    "in which jurisdictions? Write a short report with a chart."
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
    """Download the database and load it into `project/context`."""
    # In this interpreter: the project's environment has what it needs too.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        shared / "prepare_data.py",
        project / "data",
        project / "context" / "offshore_leaks.duckdb",
        cwd=shared,
    )
    if await proc.wait() != 0:
        sys.exit(f"preparing the data: exit status {proc.returncode}")


async def main(prompt: str) -> None:
    for name in ["context", "artifacts"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)
    await prepare(SHARED, PROJECT)

    # The virtx server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none. The sandboxes it holds go when this program does.
    client = Client.spawn()

    # The image, built once: Python, and what the agent's code reads the data and draws with.
    # The server answers a tag in its full form, as `docker.io/library/ailoy-offshore-leaks:1`.
    images = await client.list_image()
    if not any(image.reference.split("/")[-1] == IMAGE for image in images):
        print(f"Building {IMAGE}, once; it takes a few minutes.")
        recipe = Recipe("python:3.12-slim-trixie").run(
            # The DuckDB `prepare_data.py` writes the file with (pinned in pyproject.toml);
            # an older one may not read it.
            "pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx"
        )
        async for entry in client.build_image(recipe, IMAGE):
            if isinstance(entry.entry, ImageJobEnd) and entry.entry.error:
                raise RuntimeError(f"building {IMAGE}: {entry.entry.error}")

    # The volumes, made the first time and kept on the server: `409` says one is there already.
    volumes = []
    for name in [
        "ailoy-offshore-leaks-skill",
        "ailoy-offshore-leaks-context",
        "ailoy-offshore-leaks-artifacts",
    ]:
        try:
            volumes.append(await client.create_volume(name))
        except VirtxError as e:
            if e.status != 409:
                raise
            volumes.append(await client.get_volume(name))
    skill, context, artifacts = volumes

    # The skill onto the first.
    for name in ["SKILL.md", "oldb.py"]:
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
            SandboxVolume(
                volume=skill.name, path="/skills/offshore-leaks", readonly=True
            ),
            SandboxVolume(volume=context.name, path="/context", readonly=True),
            SandboxVolume(volume=artifacts.name, path="/artifacts"),
        ],
        network=False,
        vcpus=2,
        memory_mib=2048,
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "openai/gpt-6-astra"))
        .instruction(INSTRUCTION)
        .sandbox(sandbox)
        .skill("/skills/offshore-leaks")
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
