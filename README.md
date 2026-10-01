<p align="center">
  <picture>
    <img alt="Ailoy" src="https://brekkylab.github.io/ailoy/img/ailoy-logo-letter.png" width="352" style="max-width: 50%;">
  </picture>
</p>

<h3 align="center">AI agent builder with a VM at its heart.</h3>

<p align="center">
  <img src="https://cdn.simpleicons.org/rust/000000/ffffff" width="16"/> <a href="https://crates.io/crates/ailoy"><img src="https://img.shields.io/crates/v/ailoy?label=ailoy&color=dea584" alt="crates.io"></a>
  <img src="https://cdn.simpleicons.org/python" width="16"/> <a href="https://pypi.org/project/ailoy-py/"><img src="https://img.shields.io/pypi/v/ailoy-py?color=blue&label=ailoy-py" alt="PyPI"></a>
  <img src="https://cdn.simpleicons.org/nodedotjs" width="16"/> <a href="https://www.npmjs.com/package/ailoy-node"><img src="https://img.shields.io/npm/v/ailoy-node?label=ailoy-node&color=339933" alt="npm node"></a>
</p>

</p>
<p align="center">
  <img src="https://img.shields.io/badge/docs-coming%20soon-lightgrey" alt="Documentation coming soon">
  <!-- <a href="https://brekkylab.github.io/ailoy/"><img src="https://img.shields.io/badge/docs-eng-5a9cae" alt="Documentation"></a> -->
  <!-- <a href="https://brekkylab.github.io/ailoy/ko/"><img src="https://img.shields.io/badge/docs-kor-5a9cae" alt="Documentation"></a> -->
  <!-- <a href="https://docs.rs/ailoy"><img src="https://img.shields.io/docsrs/ailoy?label=docs.rs" alt="docs.rs"></a> -->
  <a href="https://discord.gg/27rx3EJy3P"><img src="https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/ailoy_co"><img src="https://img.shields.io/badge/X-000000?logo=x&logoColor=white" alt="X"></a>
</p>

<br>

Most agent development frameworks focuses on how to connect an LLM to tools (e.g. MCP).
Ailoy can do that too, but it also offers a way to make agents far more powerful: **give the agent a computer of its own**.

This lets you build agents that do more than call predefined tools: they can install and use software, create their own scripts, and operate in a general-purpose computing environment—without touching the host system beyond what you explicitly expose.

To make this possible, Ailoy **gives each agent a virtual machine of its own** without a separate VM daemon to install.  
Inside this isolated Linux VM, regardless of your host OS, the agent can freely:

- install packages
- run code
- work with files
- use the network
- **even run ML models on the GPU**

Ailoy works on <img src="https://cdn.simpleicons.org/linux/000000/ffffff" width="16"/> Linux, <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/windows11/windows11-original.svg" width="16"/> Windows, and <img src="https://cdn.simpleicons.org/apple/000000/ffffff" width="16"/> macOS.

> [!WARNING]
> Ailoy is under active development, and its API may change between versions.

## Requirements
**All you need is an API key** for the LLM provider you want to use for the agent.

**Nothing to install on your machine.** You don't need to install and run a VM daemon such as Docker or Kubernetes.  

The only exception is [virtx](https://github.com/brekkylab/virtx)'s virtual filesystem feature, which relies on host mounts and therefore needs FUSE support:
install [FUSE-T](https://www.fuse-t.org/) on macOS, [Dokany](https://github.com/dokan-dev/dokany) on Windows, or `fuse3` on Linux (only for non-root users; usually preinstalled).  
Without it everything else still works, and a mount fails with an error that says what to install.  
See the [virtx README](https://github.com/brekkylab/virtx) for details.

On Windows, the console's micro-VM runs on the *Windows Hypervisor Platform*, which is off by default.
Turn the optional feature on from an administrator PowerShell and restart, with virtualization enabled in the firmware:

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform
```

## Quick start

Set the API key for your model's provider, either in the environment or in a `.env` file:

```sh
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Then build your agent with this simple API in your preferred language:

<details>
<summary><b>Python</b></summary>

```sh
pip install ailoy-py
```

```python
import asyncio

from ailoy import AgentBuilder
from ailoy.virtx import ConsoleClient, Recipe


async def main() -> None:
    console = await (
        ConsoleClient.builder()
        .image(Recipe("python:3.12-slim-trixie").step("pip install matplotlib"))
        .mount("./artifacts", "/artifacts")
        .network(True)
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

`agent.run` yields one complete message for each step of the tool loop, and `agent.run_stream` streams each message token by token as the model writes it.

</details>

<details>
<summary><b>Node.js</b></summary>

```sh
npm install ailoy-node
```

```js
const { AgentBuilder, ConsoleClient, Recipe } = require('ailoy-node')

const console_ = await ConsoleClient.builder()
  .image(new Recipe('python:3.12-slim-trixie').step('pip install matplotlib'))
  .mount('./artifacts', '/artifacts')
  .network(true)
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

`agent.run` yields one complete message for each step of the tool loop, and `agent.runStream` streams each message token by token as the model writes it.

</details>

<details>
<summary><b>Rust</b></summary>

```toml
[dependencies]
ailoy = "0.3"
virtx = "0.1"
```

```rust
use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use virtx::image::Recipe;
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let console = ConsoleClient::builder()
        .image(Recipe::new("python:3.12-slim-trixie").step("pip install matplotlib"))
        .mount(std::path::absolute("./artifacts")?, "/artifacts")
        .network(true)
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

`agent.run` yields one complete message for each step of the tool loop, and `agent.run_stream` streams each message token by token as the model writes it.

</details>


...Or skip the reading: point your coding agent (Claude Code, Codex, Cursor, ...) at this README and tell it what agent you want to build.

## What can an agent do?

Rust examples are in [`examples/`](./examples) and run with `cargo run --example <name>`.

Python examples are in [`bindings/python/examples`](./bindings/python/examples).

```sh
cd bindings/python/examples/<name>
uv run main.py
```

Node examples are in [`bindings/node/examples`](./bindings/node/examples).

```sh
cd bindings/node
npm install && npm run build
node examples/<name>/main.mjs
```

> You'll need a GPU (any GPU that supports Vulkan, or Metal on Macs) for the examples that run ML models.

| Example | Description | Requirements |
| --- | --- | :---: |
| [hello](./examples/hello) | One turn with no tools and no console | |
| [cad](./examples/cad) | Writes CadQuery, renders the model from four sides, looks at the renders and iterates | |
| [offshore_leaks](./examples/offshore_leaks) | Analyses the ICIJ Offshore Leaks database with SQL and Python that the agent writes itself | Python (with uv) |
| [retail_bench](./examples/retail_bench) | Runs a supermarket simulator, one day per turn | Python (with uv) |
| [sam3](./examples/sam3) | Segments images and videos with SAM3 on the guest GPU (ncnn + Vulkan) | GPU, Python (with uv) |
| [tts](./examples/tts) | Speaks text in a voice described in words, using Qwen3-TTS | GPU, Python (with uv) |
| [laya](./examples/laya) | Answers typed decision questions with a local model on the GPU | GPU, Python (with uv) |

## Building from source

Ailoy builds on [virtx](https://github.com/brekkylab/virtx); see its README for what it needs on your host.

```sh
git clone https://github.com/brekkylab/ailoy
cd ailoy
cargo build
```

For the bindings:

```sh
cd bindings/python && uv run maturin develop  # Python
cd bindings/node && npm install && npm run build  # Node.js
```

## License

Apache-2.0. See [LICENSE.md](./LICENSE.md).
