<p align="center">
  <picture>
    <img alt="Ailoy" src="https://brekkylab.github.io/ailoy/img/ailoy-logo-letter.png" width="352" style="max-width: 50%;">
  </picture>
</p>

<p align="center">
  <img src="https://cdn.simpleicons.org/python" width="16"/> <a href="https://pypi.org/project/ailoy-py/"><img src="https://img.shields.io/pypi/v/ailoy-py?color=blue&label=ailoy-py" alt="PyPI"></a>
  <img src="https://cdn.simpleicons.org/nodedotjs" width="16"/> <a href="https://www.npmjs.com/package/ailoy-node"><img src="https://img.shields.io/npm/v/ailoy-node?label=ailoy-node&color=339933" alt="npm node"></a>
  <img src="https://cdn.simpleicons.org/webassembly" width="16"/> <a href="https://www.npmjs.com/package/ailoy-web"><img src="https://img.shields.io/npm/v/ailoy-web?label=ailoy-web&color=654ff0" alt="npm web"></a>
</p>

</p>
<p align="center">
  <a href="https://brekkylab.github.io/ailoy/"><img src="https://img.shields.io/badge/docs-eng-5a9cae" alt="Documentation"></a>
  <a href="https://brekkylab.github.io/ailoy/ko/"><img src="https://img.shields.io/badge/docs-kor-5a9cae" alt="Documentation"></a>
  <a href="https://discord.gg/27rx3EJy3P"><img src="https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/ailoy_co"><img src="https://img.shields.io/badge/X-000000?logo=x&logoColor=white" alt="X"></a>
</p>

<br>

Most agent development frameworks focuses on how to connect an LLM to tools (e.g. MCP).
Ailoy can do that too, but it also offers a way to make agents far more powerful: **give the agent a computer of its own**.

This lets you build agents that do more than call predefined tools: they can install and use software, create their own scripts, and operate in a general-purpose computing environment—without touching the host system beyond what you explicitly expose.

To make this possible, Ailoy **gives each agent a virtual machine of its own**.
Inside this isolated Linux VM, the agent can freely install packages, run code, work with files, use the network, and even run ML models on the GPU, regardless of your host OS.

This is how modern agents (ChatGPT, Claude, etc.) actually work: instead of relying only on predefined tools, they operate inside general-purpose computing environments.
**Ailoy brings that architecture to your own applications**.

Ailoy works on <img src="https://cdn.simpleicons.org/linux/000000/ffffff" width="16"/> Linux, <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/windows11/windows11-original.svg" width="16"/> Windows, and <img src="https://cdn.simpleicons.org/apple/000000/ffffff" width="16"/> macOS.

> [!WARNING]
> Ailoy is under active development, and its API may change between versions.

## Requirements

Basically **none**.

You don't need to install and run a VM daemon such as Docker.

The only exception is cortex's virtual filesystem feature, which relies on host mounts and therefore needs FUSE support:
install [FUSE-T](https://www.fuse-t.org/) on macOS or [Dokany](https://github.com/dokan-dev/dokany) on Windows.
See the [cortex README](https://github.com/brekkylab/cortex) for details.

## Quick start

Set the API key for your model's provider, either in the environment or in a `.env` file:

```sh
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

<details>
<summary><b>Python</b></summary>

```sh
pip install ailoy-py
```

```python
import asyncio

from ailoy import AgentBuilder
from ailoy.cortex import ConsoleClient, NetworkAccess, Recipe


async def main() -> None:
    console = await (
        ConsoleClient.builder()
        .image(Recipe("python:3.12-slim-trixie").step("pip install matplotlib"))
        .mount("./artifacts", "/artifacts")
        .network(NetworkAccess.public())
        .build()
    )

    agent = await (
        # For openai, use "openai/gpt-5.6-luna"
        AgentBuilder("anthropic/claude-haiku-4-5")
        .instruction("Write what you are asked for into /artifacts.")
        .system_tools()
        .console(console)
        .build()
    )

    async for output in agent.run("Create a bar chart comparing the populations of European countries and save it to /artifacts/population.png."):
        for part in output["message"]["contents"]:
            if part["type"] == "text":
                print(part["text"])

asyncio.run(main())
```

</details>

<details>
<summary><b>Node.js</b></summary>

```sh
npm install ailoy-node
```

```js
const { AgentBuilder, ConsoleClient, NetworkAccess, Recipe } = require('ailoy-node')

const console_ = await ConsoleClient.builder()
  .image(new Recipe('python:3.12-slim-trixie').step('pip install matplotlib'))
  .mount('./artifacts', '/artifacts')
  .network(NetworkAccess.public())
  .build()
// For openai, use "openai/gpt-5.6-luna"
const agent = await new AgentBuilder('anthropic/claude-haiku-4-5')
  .instruction('Write what you are asked for into /artifacts.')
  .systemTools()
  .console(console_)
  .build()

try {
  for await (const { message } of agent.run('Create a bar chart comparing the populations of European countries and save it to /artifacts/population.png.')) {
    for (const part of message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
} finally {
  await agent.close()
}
```

</details>

<details>
<summary><b>Rust</b></summary>

```toml
[dependencies]
ailoy = "0.3"
cortex = { git = "https://github.com/brekkylab/cortex" }
```

```rust
use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use cortex::{image::Recipe, protocol::NetworkAccess};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let console = ConsoleClient::builder()
        .image(Recipe::new("python:3.12-slim-trixie").step("pip install matplotlib"))
        .mount(std::path::absolute("./artifacts")?, "/artifacts")
        .network(NetworkAccess::public())
        .build()
        .await?;

    // For openai, use "openai/gpt-5.6-luna"
    let mut agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
        .instruction("Write what you are asked for into /artifacts.")
        .system_tools()
        .console(console)
        .build()
        .await?;

    let query = Message::new(Role::User).with_contents([Part::text("Create a bar chart comparing the populations of European countries and save it to /artifacts/population.png.")]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let message = output?.message;
        if message.role == Role::Assistant {
            for text in message.contents.iter().filter_map(Part::as_text) {
                println!("{text}");
            }
        }
    }
    Ok(())
}
```

</details>

`agent.run` yields one complete message for each step of the tool loop, and `run_stream` yields token deltas as the model writes them.

## What can a agent do?

Rust examples are in [`examples/`](./examples) and run with `cargo run --example <name>`.

> You'll need a GPU (any GPU that supports Vulkan, or Metal on Macs) for the examples that run ML models.

| Example | Description | Requirements |
| --- | --- | :---: |
| [hello](./examples/hello) | One turn with no tools and no console | |
| [cad](./examples/cad) | Writes CadQuery, renders the model from four sides, looks at the renders and iterates | |
| [offshore_leaks](./examples/offshore_leaks) | Analyses the ICIJ Offshore Leaks database with SQL and Python that the agent writes itself | |
| [retail_bench](./examples/retail_bench) | Runs a supermarket simulator, one day per turn | |
| [sam3](./examples/sam3) | Segments images and videos with SAM3 on the guest GPU (ncnn + Vulkan) | GPU |
| [tts](./examples/tts) | Speaks text in a voice described in words, using Qwen3-TTS | GPU |
| [laya](./examples/laya) | Answers typed decision questions with a local model on the GPU | GPU |

Python examples are in [`bindings/python/examples`](./bindings/python/examples), and Node examples are in [`bindings/node/examples`](./bindings/node/examples).

## Concepts

### Models and providers

The prefix of a model name selects its provider.
The default providers are read from the environment the first time an agent is built:

| Prefix | Environment variable | Example |
| --- | --- | --- |
| `openai/` | `OPENAI_API_KEY` | `openai/gpt-5.6-luna` |
| `anthropic/` | `ANTHROPIC_API_KEY` | `anthropic/claude-haiku-4-5` |
| `google/` | `GEMINI_API_KEY` | `google/gemini-3-pro` |
| `x-ai/` | `XAI_API_KEY` | `x-ai/grok-4` |
| `deepseek/` | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| `moonshotai/` | `KIMI_API_KEY` | `moonshotai/kimi-k2` |
| `openrouter/` | `OPENROUTER_API_KEY` | `openrouter/<vendor>/<model>` |
| `bedrock/` | `AWS_BEARER_TOKEN_BEDROCK` (+ `AWS_REGION`) | `bedrock/global.openai.gpt-6-astra` |

To use another endpoint, register it under a pattern of your own with one of the wire formats: OpenAI Responses, Chat Completions, Anthropic Messages, Gemini, or Bedrock Converse.
In Rust this is done through `LangModelProvider`, and in Python and Node through `register_lang_model` / `registerLangModel`.

### The console

A console is a cortex VM.
The guest is always Linux, and the host can be Linux, macOS on Apple silicon, or Windows 11.
The console's image is built from a recipe (a base image plus setup steps), and host folders are mounted into it read-only or writable.
The agent starts the VM for each batch of tool calls and stops it afterwards.

```python
ConsoleClient.builder()
    .image(Recipe("python:3.12-slim-trixie").step("pip install duckdb"))
    .mount_readonly("./context", "/context")
    .mount("./artifacts", "/artifacts")
    .network(NetworkAccess.public())
    .gpu(True)           # Vulkan inside the guest
    .vcpus(4).memory_mib(4096)
    .build()
```

The console is shared with the agent rather than handed over to it, so your own code can keep calling `exec`, `read` and `write` on it between turns.
Sub-agents share their parent's console.

### Tools

| Tool | What it does |
| --- | --- |
| `shell` | Runs a command in the console |
| `read`, `read_file` | Read a file, in Claude style or Gemini style (line ranges) |
| `write`, `edit`, `apply_patch` | Overwrite a file, replace an exact string, apply a Codex-style patch |
| `imgread` | Shows an image (PNG/JPEG/GIF/WEBP) to the model |
| `web_search` | Searches across Bing, Brave, DuckDuckGo, Google, Mojeek, Naver, Startpage, Yahoo and Yandex |
| `web_fetch` | Fetches pages as Markdown, respecting robots.txt |

`system_tools()` picks the file tools for the model.
`shell_tool()`, `web_search_tool()` and `web_fetch_tool()` add tools one at a time.

Custom tools can be added as well:

```python
def temperature(city: str) -> float:
    return 21.5

desc = ailoy.register_tool(
    {
        "name": "temperature",
        "description": "The temperature in a city now, in Celsius.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    temperature,
)
agent = await ailoy.AgentBuilder("anthropic/claude-haiku-4-5").tool(desc).build()
```

MCP servers are added with `register_mcp_stdio` / `register_mcp_streamable_http`, and A2A agents with `register_a2a`.
Each of these returns the tool descriptions to pass to the builder.
A tool that raises an error returns the error to the model instead of ending the turn.

### Skills, sub-agents and memory

- **Skills**: `.skill("/skills/cad")` points to a folder in the console that contains a `SKILL.md`. Its frontmatter is listed in the system message, and the agent reads the full file when it uses the skill.
- **Sub-agents**: an `AgentSpec` with an `AgentCard` becomes a tool that the parent agent can call.
- **Memory**: a `Memory` store gives the agent `mem_search` and `mem_insert` tools, backed by a single file in the console.
- **Context management**: when the input grows past a token limit, the oldest turns are dropped, and the system message and the most recent turns are kept.

## Building from source

Ailoy links [cortex](https://github.com/brekkylab/cortex) from a checkout next to this repository:

```
workspace/
├── ailoy/
└── cortex/
```

On macOS, cortex's host mounts require FUSE-T: `brew install --cask fuse-t`.

```sh
cargo build                                   # Rust crate
cargo run --example hello -- anthropic/claude-haiku-4-5

cd bindings/python && uv sync && uv run maturin develop   # Python
cd bindings/node && npm install && npm run build          # Node
```

For details, see [bindings/python/README.md](./bindings/python/README.md) and [bindings/node/README.md](./bindings/node/README.md).

## Requirements

| | |
| --- | --- |
| Rust | 1.95+ (edition 2024) |
| Python | 3.10+ |
| Node.js | 18+ |
| Host OS | Linux, macOS (Apple silicon), Windows 11 |

## License

Apache-2.0. See [LICENSE.md](./LICENSE.md).
