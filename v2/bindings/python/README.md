# ailoy-py

Ailoy is a lightweight library for building AI applications — such as **agent systems** or **RAG pipelines** — with ease. It is designed to enable AI features effortlessly, one can just import and use.

See our [documentation](https://brekkylab.github.io/ailoy) for more details.

## Install

```bash
pip install ailoy-py
```

## Quickstart

```python
import asyncio

from ailoy import Agent, LocalLanguageModel

async def main():
    # Create Qwen3-0.6B LocalLanguageModel
    async for v in LocalLanguageModel.create("Qwen/Qwen3-0.6B"):
        if v.result:
            model = v.result

    # Create an agent using this model
    agent = Agent(model, [])

    # Ask a prompt and iterate over agent's responses
    async for resp in agent.run("What is your name?"):
        print(resp)


if __name__ == "__main__":
    asyncio.run(main())
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
pip install -e .
```

### Generate wheel

```bash
python -m build -w
```
