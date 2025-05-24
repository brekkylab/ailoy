# Ailoy

Ailoy is a lightweight library for building AI applications — such as **agent systems** or **RAG pipelines** — with ease. It is designed to enable AI features effortlessly, one can just import and use.

> [!WARNING]
> This library is in an early stage of development. APIs may change without notice.

## Supports

Ailoy supports the following operating systems:
- Windows (x86, with Vulkan)
- macOS (Apple Silicon, with Metal)
- Linux (x86, with Vulkan)

Currently, the following AI models are supported:
- Qwen/Qwen3-0.6B (on-device)
- Qwen/Qwen3-1.7B (on-device)
- Qwen/Qwen3-4B (on-device)
- Qwen/Qwen3-8B (on-device)
- gpt-4o (API key needed)

## Getting Started

### Node

```typescript
import {
  startRuntime,
  createAgent,
} from "ailoy-js-node";

const rt = await startRuntime();
const agent = await createAgent(rt, {model: {name: "Qwen/Qwen3-0.6B"}});
for await (const resp of agent.query("When is your cut-off date?")) {
  agent.print(resp);
}
await agent.delete();
await rt.stop();
```

For more details, refer to `bindings/js-node/README.md`.

### Python

```python
from ailoy import Runtime, Agent

rt = Runtime()
agent = Agent(rt, model_name="Qwen/Qwen3-8B")
for resp in agent.query("When is your cut-off date?"):
    resp.print()
agent.delete()
rt.stop()
```

For more details, refer to `bindings/python/README.md`.

## Build from source

### Prerequisites

- C/C++ compiler
  (recommended versions are below)
  - GCC >= 13
  - LLVM Clang >= 17
  - Apple Clang >= 15
  - MSVC >= 19.29
- CMake >= 3.24.0
- Git
- OpenSSL (libssl-dev)
- Rust & Cargo >= 1.82.0 (optional, required for mlc-llm)
- OpenMP (libomp-dev) (optional, used by Faiss)
- BLAS (libblas-dev) (optional, used by Faiss)
- LAPACK (liblapack-dev) (optional, used by Faiss)

### Node.js Build

```bash
cd bindings/js-node
npm run build
```

For more details, refer to `bindings/js-node/README.md`.

### Python Build

```bash
cd bindings/python
pip wheel --no-deps -w dist .
```

For more details, refer to `bindings/python/README.md`.
