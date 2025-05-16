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
  createAgent,
} from "ailoy-js-node";

const rt = await startRuntime();
const ex = await createAgent(rt, {model: {name: "qwen3-0.6b"}});
for await (const resp of ex.run("When is your cut-off date?")) {
    console.log(resp);
}
```

For more details, refer to `bindings/js-node/README.md`.

### Python

```python
from ailoy import AsyncRuntime, Agent

rt = AsyncRuntime()
ex = Agent(rt, model_name="qwen3-8b")
async for resp in ex.run("When is your cut-off date?"):
    print(resp)
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

<details>

<summary>Example commands</summary>
<h3>Ubuntu 24.04</h3>

```bash
# APT packages
apt-get install -y git wget curl
apt-get install -y libblas-dev liblapack-dev libssl-dev  # BLAS / LAPACK / OpenSSL
apt-get install -y build-essential                       # g++13

# Rust
curl https://sh.rustup.rs -sSf | sh

# CMake
bash -c "$(wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-aarch64.sh)"

# Clone Ailoy
git clone https://github.com/brekkylab/ailoy && cd ailoy

# Build from source (can be replaced by Node.js/Python builds below)
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
</details>


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
