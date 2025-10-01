# ailoy-py

Ailoy is a lightweight library for building AI applications — such as **agent systems** or **RAG pipelines** — with ease. It is designed to enable AI features effortlessly, one can just import and use.

See our [documentation](https://brekkylab.github.io/ailoy) for more details.

## Install

```bash
pip install ailoy-py
```

## Quickstart

### Asynchronous version (recommended)
```python
import asyncio

from ailoy import Agent, LocalLanguageModel

async def main():
    # Create Qwen3-0.6B LocalLanguageModel
    model = await LocalLanguageModel.create("Qwen/Qwen3-0.6B")

    # Create an agent using this model
    agent = Agent(model)

    # Ask a prompt and iterate over agent's responses
    async for resp in agent.run("What is your name?"):
        print(resp)


if __name__ == "__main__":
    asyncio.run(main())
```

### Synchronous version
```python
from ailoy import Agent, LocalLanguageModel

def main():
    # Create Qwen3-0.6B LocalLanguageModel
    model = await LocalLanguageModel.create_sync("Qwen/Qwen3-0.6B")

    # Create an agent using this model
    agent = Agent(model)

    # Ask a prompt and iterate over agent's responses
    for resp in agent.run_sync("What is your name?"):
        print(resp)


if __name__ == "__main__":
    main()
```

## Building from source

### Prerequisites

- Rust >= 1.88
- Python >= 3.10
- C/C++ compiler
  (recommended versions are below)
  - GCC >= 13
  - LLVM Clang >= 17
  - Apple Clang >= 15
  - MSVC >= 19.29
- CMake >= 3.28.0
- Git
- OpenMP (required to build Faiss)
- BLAS (required to build Faiss)
- LAPACK (required to build Faiss)
- Vulkan SDK (on Windows and Linux)


### Setup development environment

```bash
pip install maturin

# This generates `_core.cpython-3xx-darwin.so` under `ailoy/`
maturin develop
```

### Generate wheel

```bash
maturin build --out ./dist
```
