"""Analyze ICIJ's Offshore Leaks database with an agent that writes and runs its own SQL and
Python, in a console, against the data it is given as context.

    uv run main.py
    uv run main.py "Which South Korean officers appear in more than one leak?"

The Python side of the Rust `offshore_leaks` example in `examples/offshore_leaks`, whose
`main.rs` has the long form. `prepare_data.py` loads ICIJ's CSVs into one DuckDB file in
`context/` on the first run, which the agent queries in place. The skill is `SKILL.md` and
`oldb.py`, mounted at `/skills/offshore-leaks` from memory. This folder is a uv project of its
own (`pyproject.toml`), which `uv run` sets up with ailoy from this checkout; run it from here.

* `context/` at `/context`, read-only — `offshore_leaks.duckdb`, and whatever else the
  request is about, such as a list of names to look for.
* `artifacts/` at `/artifacts`, writable — where the reports, tables and charts go.

The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA.
Being in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.

Environment:

* `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads; ICIJ's latest by default.
* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its
  provider's API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).

Read from `.env` as well.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Absolute, because a mount is named to the server as a `file://` URL.
HERE = Path(__file__).resolve().parent

# This folder is a uv project of its own. Started from another environment, as with
# `uv run python examples/offshore_leaks/main.py` from `bindings/python`, run again in this one.
if Path(sys.prefix).resolve() != HERE / ".venv" and "AILOY_EXAMPLE_REEXEC" not in os.environ:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["AILOY_EXAMPLE_REEXEC"] = "1"  # once: not again if uv puts the environment elsewhere
    os.execvpe("uv", ["uv", "run", "--directory", str(HERE), "main.py", *sys.argv[1:]], env)

import ailoy  # noqa: E402
from ailoy.virtx import ConsoleClient, Directory, HostMount, Recipe  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

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


async def prepare(project: Path) -> None:
    """Download the database and load it into `project/context`."""
    # In this interpreter: the project's environment has what it needs too.
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "prepare_data.py", cwd=project
    )
    if await proc.wait() != 0:
        sys.exit(f"preparing the data: exit status {proc.returncode}")


async def main(prompt: str) -> None:
    # `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for name in ["context", "artifacts", "skill"]:
        (HERE / name).mkdir(parents=True, exist_ok=True)
    await prepare(HERE)

    console = await (
        ConsoleClient.builder()
        .image(
            Recipe("python:3.12-slim-trixie").step(
                # The DuckDB `prepare_data.py` writes the file with (pinned in pyproject.toml);
                # an older one may not read it.
                "pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx"
            )
        )
        .mount_readonly(
            HostMount(
                Directory()
                .with_file("SKILL.md", (HERE / "SKILL.md").read_bytes())
                .with_file("oldb.py", (HERE / "oldb.py").read_bytes()),
                HERE / "skill",
            ),
            "/skills/offshore-leaks",
        )
        .mount_readonly(HERE / "context", "/context")
        .mount(HERE / "artifacts", "/artifacts")
        .network(False)
        .vcpus(2)
        .memory_mib(2048)
        .build()
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "openai/gpt-6-astra"))
        .instruction(INSTRUCTION)
        .system_tools()
        .console(console)
        .skill("/skills/offshore-leaks")
        .build()
    )

    async with console:
        async for output in agent.run(prompt):
            message = output["message"]
            if message["role"] == "assistant":
                for part in message["contents"]:
                    if part["type"] == "text":
                        print(part["text"])
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    print(f"→ {function['name']} {json.dumps(function['arguments'], indent=2)}")
            elif message["role"] == "tool":
                for part in message["contents"]:
                    print(f"← {json.dumps(part, indent=2, ensure_ascii=False, default=_bytes)}")
            sys.stdout.flush()


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:]) or QUERY))
