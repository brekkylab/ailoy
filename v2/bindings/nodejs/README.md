# ailoy-node

Ailoy is a lightweight library for building AI applications — such as **agent systems** or **RAG pipelines** — with ease. It is designed to enable AI features effortlessly, one can just import and use.

See our [documentation](https://brekkylab.github.io/ailoy) for more details.

## Install

```bash
# npm
npm install ailoy-node
# yarn
yarn add ailoy-node
```

## Quickstart

```typescript
import * as ailoy from "ailoy-node";

async function main() {
    // Create Qwen3-0.6B LocalLanguageModel
    const model = await ailoy.LocalLanguageModel.create("Qwen/Qwen3-0.6B");

    // Create an agent using this model
    const agent = new ailoy.Agent(model);

    // Ask a prompt and iterate over agent's responses
    for await (const resp of agent.run("What is your name?")) {
        console.log(resp)
    }
}

await main()
```

## Building from source

### Prerequisites

- Rust >= 1.88
- Node.js >= LTS version
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
# Install dev dependencies
npm install

# This generates `ailoy_core.<platform>.node` under `src/`
npm run build:napi

# This builds the bundled index.js and index.d.ts under `dist/`
npm run build:ts

# To build napi module and typescript at once
npm run build
```

### Tests

Some test cases require specific environment variables to be set before running the test (e.g. `OPENAI_API_KEY`). Please fill `.env.template` and rename to `.env` in order to run every tests.

```bash
# Run all test cases
npm run test

# Run test cases with filtering
npm run test -- -t "Agent.*Local"
```

### Generate npm package

```bash
npm pack
```
