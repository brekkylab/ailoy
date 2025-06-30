# Ailoy

[![document](https://img.shields.io/badge/document-latest-2ea44f)](https://brekkylab.github.io/ailoy/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ailoy-py)](https://pypi.org/project/ailoy-py/)
[![PyPI - Python Version](https://img.shields.io/pypi/v/ailoy-py)](https://pypi.org/project/ailoy-py/)
[![NPM Version](https://img.shields.io/npm/v/ailoy-node)](https://www.npmjs.com/package/ailoy-node)

[![Discoard](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/27rx3EJy3P)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Failoy_co)](https://x.com/ailoy_co)


Ailoy is a lightweight library for building AI applications — such as **agent systems** or **RAG pipelines** — with ease. It is designed to enable AI features effortlessly, one can just import and use.

> [!WARNING]
> This library is in an early stage of development. APIs may change without notice.

> [!TIP]
> We have a [Discord channel](https://discord.gg/27rx3EJy3P)! If you get stuck or have any questions, feel free to join and ask.

> [!TIP]
> There are several interesting examples in the [`examples`](./examples) directory — take a look!
>
> You might get inspired what kind of applications you can build with Agents.

## Features

- Run AI models either in local or via cloud APIs
- Multi-turn conversation and system message customization
- Support for reasoning-based workflows
- Tool calling capabilities (including `MCP` integration)
- Built-in vector store support (via `Faiss` or `ChromaDB`)

For more details, please refer to the [documentation](https://brekkylab.github.io/ailoy/).

Currently, the following AI models are supported:
- Language Models
  - Local Models
    - Qwen/Qwen3-0.6B
    - Qwen/Qwen3-1.7B
    - Qwen/Qwen3-4B
    - Qwen/Qwen3-8B
    - Qwen/Qwen3-14B
    - Qwen/Qwen3-32B
    - Qwen/Qwen3-30B-A3B
  - API Models
    - OpenAI
    - Gemini
    - Claude
- Embedding Models
  - Local Models
    - BAAI/bge-m3

You can check out examples for a simple chatbot, tool usages and retrieval-augmented generation (RAG).

## Requirements

Ailoy supports the following operating systems:
- Windows (x86_64, with Vulkan)
- macOS (Apple Silicon, with Metal)
- Linux (x86_64, with Vulkan)

To use Ailoy with local models, a compatible device is required.
However, if your system doesn't meet the hardware requirements, you can still run Ailoy using external APIs such as OpenAI.

AI models typically consume a significant amount of memory.
The exact usage depends on the model size, but we recommend at least **8GB of GPU memory**.
Running the Qwen 8B model requires at least **12GB of GPU memory**.
On macOS, this refers to unified memory, as Apple Silicon uses a shared memory architecture.

### For running AI locally

**Windows**
- CPU: Intel Skylake or newer (and compatible AMD), x86_64 is required
- GPU: At least 8GB of VRAM and support for Vulkan 1.3
- OS: Windows 11 or Windows Server 2022 (earlier versions may work but are not officially tested)
- NVIDIA driver that supports Vulkan 1.3 or higher

**macOS**
- Device: Apple Silicon with Metal support
- Memory: At least 8GB of unified memory
- OS: macOS 14 or newer

**Linux**
- CPU: Intel Skylake or newer (and compatible AMD), x86_64 is required
- GPU: At least 8GB of VRAM and support for Vulkan 1.3
- OS: Debian 10 / Ubuntu 21.04 or newer (this means, os with glibc 2.28 or higher)
- NVIDIA driver that supports Vulkan 1.3 or higher

## Getting Started

### Node

```sh
npm install ailoy-node
```

```typescript
import {
  startRuntime,
  defineAgent,
  LocalModel
} from "ailoy-node";

(async () => {
  const rt = await startRuntime();
  const agent = await defineAgent(rt, LocalModel({id: "Qwen/Qwen3-0.6B"}));
  for await (const resp of agent.query("Hello world!")) {
    agent.print(resp);
  }
  await agent.delete();
  await rt.stop();
})();
```

### Python

```sh
pip install ailoy-py
```

```python
from ailoy import Runtime, Agent, LocalModel

rt = Runtime()
with Agent(rt, LocalModel("Qwen/Qwen3-0.6B")) as agent:
    for resp in agent.query("Hello world!"):
        resp.print()
rt.stop()
```

## Build from source

### Node.js Build

```bash
cd bindings/js-node
npm run build
```

For more details, refer to `bindings/js-node/README.md`.

### Python Build

```bash
cd bindings/python
pip install -e .
```

For more details, refer to `bindings/python/README.md`.
