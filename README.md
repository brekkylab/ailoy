# Ailoy

[![document](https://img.shields.io/badge/document-latest-2ea44f?color=5a9cae)](https://brekkylab.github.io/ailoy/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ailoy-py)](https://pypi.org/project/ailoy-py/)
[![PyPI - Python Version](https://img.shields.io/pypi/v/ailoy-py?color=blue)](https://pypi.org/project/ailoy-py/)
[![NPM Version](<https://img.shields.io/npm/v/ailoy-node?label=npm(node)&color=339933>)](https://www.npmjs.com/package/ailoy-node)
[![NPM Version](<https://img.shields.io/npm/v/ailoy-node?label=npm(web)&color=654ff0>)](https://www.npmjs.com/package/ailoy-node)

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/27rx3EJy3P)
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
- Supports running _native-equivalent_ functionality in the **web browsers(WASM)**

For more details, please refer to the [documentation](https://brekkylab.github.io/ailoy/).

### Supported AI models

Currently, the following AI models are supported:

- Language Models
  - Local Models
    - <img src="https://github.com/user-attachments/assets/177461a2-7a0e-4449-b5a0-8d7028349607" width="20" height="20"> Qwen3
      <details>
        <summary>Click to see details</summary><p>
          
        - `Qwen/Qwen3-0.6B`
        - `Qwen/Qwen3-1.7B`
        - `Qwen/Qwen3-4B`
        - `Qwen/Qwen3-8B`
        - `Qwen/Qwen3-14B`
        - `Qwen/Qwen3-32B`
        - `Qwen/Qwen3-30B-A3B` (MoE)
        </p>
      </details>
  - API Models
    - <img src="https://github.com/user-attachments/assets/ffc93fe4-a345-4525-bf19-0f1419af08f8" width="20" height="20"> OpenAI
    - <img src="https://github.com/user-attachments/assets/6fc0015d-090d-4642-a056-3fbc1f66b599" width="25" height="25"> Gemini
    - <img src="https://github.com/user-attachments/assets/94855f00-a640-40e2-b3b6-9481d6bfd910" width="20" height="20"> Claude
    - <img src="https://github.com/user-attachments/assets/aaf28fe3-9b1e-479d-9631-986afc8b5b66" width="20" height="20"> Grok
- Embedding Models
  - Local Models
    - <img src="https://bge-model.com/_static/bge_logo.jpeg" width="20" height="20"> BAAI/bge-m3

You can check out examples for a simple chatbot, tool usages and retrieval-augmented generation (RAG).

## Requirements

### For Agents with LLM APIs

You can create _your own agent_ with **Ailoy** using external APIs such as OpenAI, Claude, Gemini or Grok.  
Your system doesn't need to meet the hardware requirements with these APIs.

- <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/22661c5c-43a1-4ab8-9281-a9b5ad43cefb" /> Windows (x86_64)
- <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/1e77e2be-4a6e-42a5-866d-e4bd1b5a7bb7" /> macOS (Apple Silicon)
- <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/1877d447-66e7-4796-994c-d0242475a1d0" /> Linux (x86_64)
- 🌐 **Web Browsers**

### For Agents with AI running locally

To use Ailoy with local models, a compatible device is required.  
If you can use a _compatible_ device,

**Ailoy** supports _Local AI executions_ on the following environments(details are [below](#local-ai-requirement-details).):

- <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/22661c5c-43a1-4ab8-9281-a9b5ad43cefb" /> Windows (x86_64, with <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/2dde8002-f2fb-4ed1-a83c-9dd3b8a2b8bd" /> Vulkan)
- <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/1e77e2be-4a6e-42a5-866d-e4bd1b5a7bb7" /> macOS (Apple Silicon, with <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/0bf3a719-b221-4d05-8a9e-3a37673a401d" />
  Metal)
- <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/1877d447-66e7-4796-994c-d0242475a1d0" /> Linux (x86_64, with <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/2dde8002-f2fb-4ed1-a83c-9dd3b8a2b8bd" /> Vulkan)
- 🌐 **Modern Web Browsers** (wasm32, with <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/74ac8877-0265-48fe-9939-a256ce79d2d8" /> WebGPU)
  - <img src="https://github.com/user-attachments/assets/4cc0058b-92c0-4cb2-8179-3d5416ed3196" width="20" height="20"> Chrome browsers for PC/Mac is recommended.
  - Supports up to Qwen3-8B

AI models typically consume a significant amount of memory.  
The exact usage depends on the model size, but we recommend at least **8GB of GPU memory**.  
Running the Qwen 8B model requires at least **12GB of GPU memory**.  
On macOS, this refers to unified memory, as Apple Silicon uses a shared memory architecture.

#### Local AI requirement details

**Windows**

- CPU: Intel Skylake or newer (and compatible AMD), x86_64 is required
- GPU: At least 8GB of VRAM and support for Vulkan 1.3
  - NVIDIA/AMD/Intel graphic driver that supports Vulkan 1.3 or higher
- OS: Windows 11 or Windows Server 2022 (earlier versions may work but are not officially tested)

**macOS**

- Device: Apple Silicon with Metal support
- Memory: At least 8GB of unified memory
- OS: macOS 14 or newer

**Linux**

- CPU: Intel Skylake or newer (and compatible AMD), x86_64 is required
- GPU: At least 8GB of VRAM and support for Vulkan 1.3
  - NVIDIA/AMD/Intel graphic driver that supports Vulkan 1.3 or higher
- OS: Debian 10 / Ubuntu 21.04 or newer (this means, os with glibc 2.28 or higher)

**Web browser**

- Browser: Modern browsers with WebGPU support
- GPU: WebGPU with `"shader-f16"` support
  - It may not work on mobile devices with Qualcomm APs.

## Getting Started

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

### Node

```sh
npm install ailoy-node
```

```typescript
import { startRuntime, defineAgent, LocalModel } from "ailoy-node";

(async () => {
  const rt = await startRuntime();
  const agent = await defineAgent(rt, LocalModel({ id: "Qwen/Qwen3-0.6B" }));
  for await (const resp of agent.query("Hello world!")) {
    agent.print(resp);
  }
  await agent.delete();
  await rt.stop();
})();
```

### Web browser

```sh
npm install ailoy-web
```

```typescript
import * as ai from "ailoy-web";

const rt = await ai.startRuntime();
const agent = await ai.defineAgent(
  rt,
  ai.LocalModel({ id: "Qwen/Qwen3-0.6B" })
);

(async () => {
  // Assume that `textarea` with id "answer" exists
  const textarea = document.getElementById("answer");

  for await (const resp of agent.query("Hello, world!")) {
    textarea.innerHTML += resp.content;
  }
})();
```

## Build from source

### Python Build

```bash
cd bindings/python
pip install -e .
```

For more details, refer to `bindings/python/README.md`.

### Node.js Build

```bash
cd bindings/js-node
npm run build
```

For more details, refer to `bindings/js-node/README.md`.

### Javascript Build for web browsers

```bash
cd bindings/js-web
npm run build
```

For more details, refer to `bindings/js-web/README.md`.
