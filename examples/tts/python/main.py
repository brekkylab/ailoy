"""Speak a text in a voice described in words with Qwen3-TTS, through ncnn on the guest's Vulkan
device, as an agent's skill.

    uv run main.py

The Python side of the Rust `tts` example in `examples/tts/rust`, whose `main.rs` has the
long form. There is no prompt: the agent reads `text.txt` and `instruct.txt` from the context
folder, speaks the text in that voice and hands back the WAV. The skill is `SKILL.md` and
`run_tts.py`, mounted at `/skills/tts` from memory. They, `prepare_model.py`, which downloads
and converts the models into `data/` on the first run, and `context_example/` are the ones the
three sides share in `examples/tts/shared`. What a run uses is in this folder, which is a uv
project of its own (`pyproject.toml`) that `uv run` sets up with ailoy from this checkout; run
it from here.

* `context/` at `/context`, read-only — the text and the instruction; `context_example/` is
  copied in when it is empty.
* `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.

Qwen3-TTS and its weights are the Qwen team's, under Apache-2.0.

Environment:

* `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its
  provider's API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).

Read from `.env` as well.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

# Absolute, because a mount is named to the server as a `file://` URL.
HERE = Path(__file__).resolve().parent
# What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
SHARED = HERE.parent / "shared"
# What a run reads and writes, beside this file.
PROJECT = HERE

# This folder is a uv project of its own. Started from another environment, as with
# `python examples/tts/python/main.py` from the top of the checkout, run again in this one.
if Path(sys.prefix).resolve() != HERE / ".venv" and "AILOY_EXAMPLE_REEXEC" not in os.environ:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["AILOY_EXAMPLE_REEXEC"] = "1"  # once: not again if uv puts the environment elsewhere
    os.execvpe("uv", ["uv", "run", "--directory", str(HERE), "main.py", *sys.argv[1:]], env)

import ailoy  # noqa: E402
from virtx import ConsoleClient, Directory, HostMount, Recipe, ensure_virtx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

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
    # `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for name in ["context", "artifacts", "skill"]:
        (PROJECT / name).mkdir(parents=True, exist_ok=True)
    # Something to say, for a first run.
    context = PROJECT / "context"
    if not any(context.iterdir()):
        for entry in (SHARED / "context_example").iterdir():
            shutil.copy(entry, context / entry.name)

    # The console server, fetched into virtx's cache the first time: a host that installed only
    # ailoy has none.
    await ensure_virtx()
    console = await (
        ConsoleClient.builder()
        # Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
        .image(
            Recipe("python:3.12-slim-trixie")
            # `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the
            # wheel opens. Mesa from backports: 26.0 against trixie's 25.0.
            .step(
                "echo 'deb http://deb.debian.org/debian trixie-backports main' "
                "> /etc/apt/sources.list.d/backports.list "
                "&& apt-get update && apt-get install -y --no-install-recommends "
                "-t trixie-backports mesa-vulkan-drivers "
                "&& apt-get install -y --no-install-recommends libvulkan1 "
                "&& rm -rf /var/lib/apt/lists/*"
            )
            .step("pip install --no-cache-dir ncnn numpy tokenizers")
        )
        .mount_readonly(PROJECT / "data" / "ncnn", "/models")
        .mount_readonly(
            HostMount(
                Directory()
                .with_file("SKILL.md", (SHARED / "SKILL.md").read_bytes())
                .with_file("run_tts.py", (SHARED / "run_tts.py").read_bytes()),
                PROJECT / "skill",
            ),
            "/skills/tts",
        )
        .mount_readonly(context, "/context")
        .mount(PROJECT / "artifacts", "/artifacts")
        .gpu(True)
        .vcpus(2)
        # The talker is 2.8 GB on the device in fp16, and its cache and the codec's buffers
        # come on top of it there; the run itself takes under 1 GB of memory.
        .memory_mib(4096)
        .gpu_memory_mib(8192)
        .build()
    )

    agent = await (
        ailoy.AgentBuilder(os.environ.get("AILOY_MODEL", "anthropic/claude-sonnet-5"))
        .instruction(INSTRUCTION)
        .max_tokens(64000)
        .system_tools()
        .console(console)
        .skill("/skills/tts")
        .build()
    )

    async with console:
        async for output in agent.run(QUERY):
            message = output["message"]
            if message["role"] == "assistant":
                for part in message["contents"]:
                    if part["type"] == "text":
                        print(part["text"])
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    print(f"→ {function['name']} {json.dumps(function['arguments'], indent=2)}")
            # A run prints where the speech went and how long it is, and ncnn's device log
            # and its progress: shown whole.
            elif message["role"] == "tool":
                for part in message["contents"]:
                    print(f"← {json.dumps(part, indent=2, ensure_ascii=False, default=_bytes)}")
            sys.stdout.flush()


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main())
