"""Segment images and videos with SAM3 through ncnn on the guest's Vulkan device, as an
agent's skill.

    uv run main.py
    uv run main.py "Find every cat in the photos in context and mask them"

The Python side of the Rust `sam3` example in `examples/sam3`, whose `main.rs` has the long
form. The skill is `SKILL.md` and `run_sam3.py`, mounted at `/skills/sam3` from memory, which
the agent runs with its `shell` tool. `prepare_model.py` downloads and converts the model into
`data/` on the first run. This folder is a uv project of its own (`pyproject.toml`), which
`uv run` sets up with ailoy from this checkout; run it from here.

* `context/` at `/context`, read-only — the images and frames to segment, when they are not
  in the prompt.
* `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.

Environment:

* `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default. It has to be
  built with the `gpu` feature, or the session is refused with `UNSUPPORTED_MACHINE`.
* `AILOY_MODEL` — the agent's model, `bedrock/global.openai.gpt-6-astra` by default; its
  provider's API key has to be set (`AWS_BEARER_TOKEN_BEDROCK`, `ANTHROPIC_API_KEY`, ...).

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
# `uv run python examples/sam3/main.py` from `bindings/python`, run again in this one.
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

import ailoy
from ailoy.cortex import (  # noqa: E402
    Console,
    Directory,
    HostMount,
    Image,
    NetworkAccess,
)
from dotenv import load_dotenv  # noqa: E402

INSTRUCTION = (
    "# Context\n\n"
    "Path: /context\n\n"
    "Holds what you were given to segment, such as images, and videos as folders of "
    "frames. When the request refers to something that is not in it, look here first. "
    "List the folder, and pass SAM3 the paths that bear on the request. "
    "This folder is read-only.\n\n"
    "# Artifacts\n\n"
    "Path: /artifacts\n\n"
    "Where the files the user asks for go, such as the masks and overlays SAM3 writes or a "
    "report on them. Have SAM3 write into a folder here named so the user can tell what it "
    "is. A result that is only in your reply is not delivered as a file."
)


def _bytes(value: object) -> str:
    """What `json.dumps` prints for an image's bytes, as `imgread` hands them back."""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


async def prepare(project: Path) -> None:
    """Download and convert the models into `project/data`."""
    # In this interpreter: the project's environment has what it needs too.
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "prepare_model.py", cwd=project
    )
    if await proc.wait() != 0:
        sys.exit(f"preparing the models: exit status {proc.returncode}")


async def main(prompt: str) -> None:
    await prepare(HERE)
    # `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for name in ["context", "artifacts", "skill"]:
        (HERE / name).mkdir(parents=True, exist_ok=True)
    # `build()` builds the image before it starts the console, which takes a while.

    print("building the image ...", flush=True)
    console = await (
        Console.builder()
        .stdio_client([os.environ.get("AILOY_CORTEX_CONSOLE", "cortex-krun")])
        .image(
            Image()
            .base("python:3.12-slim-trixie")
            # Mesa from backports: venus passes VK_KHR_shader_bfloat16 and VK_KHR_cooperative_matrix
            # through from 26.0 on, and trixie itself has 25.0.
            .step(
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
            .step(
                "pip install --no-cache-dir av numpy opencv-python-headless pillow "
                "portalocker requests tokenizers tqdm "
                "&& pip install --no-cache-dir --no-deps ncnn"
            )
        )
        .mount_readonly(HERE / "data" / "ncnn", "/models")
        .mount_readonly(
            HostMount(
                Directory()
                .with_file("SKILL.md", (HERE / "SKILL.md").read_bytes())
                .with_file("run_sam3.py", (HERE / "run_sam3.py").read_bytes()),
                HERE / "skill",
            ),
            "/skills/sam3",
        )
        .mount_readonly(HERE / "context", "/context")
        .mount(HERE / "artifacts", "/artifacts")
        # The build's `apt-get` and `pip` run with the session's reach.
        .network(NetworkAccess.public())
        .gpu(True)
        .vcpus(2)
        .memory_mib(4096)
        .gpu_memory_mib(12288)
        .build()
    )

    agent = await (
        ailoy.AgentBuilder(
            os.environ.get("AILOY_MODEL", "bedrock/global.openai.gpt-6-astra")
        )
        .instruction(INSTRUCTION)
        .system_tools()
        .web_fetch_tool()
        .web_search_tool([])
        .console(console)
        .skill("/skills/sam3")
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


if __name__ == "__main__":
    # From the nearest `.env` up from this file, as the Rust examples load it.
    load_dotenv()
    asyncio.run(main(" ".join(sys.argv[1:])))
