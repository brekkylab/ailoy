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
- qwen3-0.6b (on-device)
- qwen3-1.7b (on-device)
- qwen3-4b (on-device)
- qwen3-8b (on-device)
- gpt-4o (API key needed)

## Getting Started

### Node

```typescript
import {
  startRuntime,
  createReflectiveExecutor,
  bearerAutenticator,
} from "ailoy-js-node";

const rt = await startRuntime();
const ex = await createReflectiveExecutor(rt, {model: {name: "qwen3-0.6b"}});
for await (const resp of ex.run("When is your cut-off date?")) {
    console.log(resp);
}
```

For more details, refer to `bindings/js-node/README.md`.

### Python

```python
from ailoy import AsyncRuntime, ReflectiveExecutor

rt = AsyncRuntime()
ex = ReflectiveExecutor(rt, model_name="qwen3-8b")
async for resp in ex.run(query):
    print_reflective_response(resp)
```

For more details, refer to `bindings/python/README.md`.

## Build from source

### Prerequisites

- C/C++ compiler
- CMake
- Git
- OpenSSL (libssl-dev)
- Rust & Cargo (optional, required for mlc-llm)
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
