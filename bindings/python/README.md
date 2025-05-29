# ailoy-py

Python binding for Ailoy runtime APIs.

See our [documentation](https://brekkylab.github.io/ailoy) for more details.

## Install

```bash
pip install ailoy-py
```

## Quickstart

```python
from ailoy import Runtime, Agent

# The runtime must be started to use Ailoy
rt = Runtime()

# Defines an agent
# During this step, the model parameters are downloaded and the LLM is set up for execution
with Agent(rt, model_name="Qwen/Qwen3-0.6B") as agent:
    # This is where the actual LLM call happens
    for resp in agent.query("Please give me a short poem about AI"):
        agent.print(resp)

# Stop the runtime
rt.stop()
```

## Building from source

### Prerequisites

- Python 3.10 or higher
- C/C++ compiler
  (recommended versions are below)
  - GCC >= 13
  - LLVM Clang >= 17
  - Apple Clang >= 15
  - MSVC >= 19.29
- CMake >= 3.28.0
- Git
- OpenSSL
- Rust & Cargo >= 1.82.0
- OpenMP
- BLAS
- LAPACK
- Vulkan SDK (if you are using vulkan)


### Setup development environment

```bash
pip install -e .
```

### Generate wheel

```bash
python -m build -w
```
