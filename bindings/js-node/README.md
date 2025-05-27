# ailoy-node

Node.js binding for Ailoy runtime APIs.

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
import { startRuntime, defineAgent } from "ailoy-node";

(async () => {
  // The runtime must be started to use Ailoy
  const rt = await startRuntime();

  // Defines an agent
  // During this step, the model parameters are downloaded and the LLM is set up for execution
  const agent = await defineAgent(rt, "Qwen/Qwen3-0.6B");

  // This is where the actual LLM call happens
  for await (const resp of agent.query("Please give me a short poem about AI")) {
    agent.print(resp);
  }

  // Once the agent is no longer needed, it can be released
  await agent.delete();

  // Stop the runtime
  await rt.stop();
})();
```

## Building from source

### Prerequisites

- Node 20 or higher
- C/C++ compiler
  (recommended versions are below)
  - GCC >= 13
  - LLVM Clang >= 17
  - Apple Clang >= 15
  - MSVC >= 19.29
- CMake >= 3.24.0
- Git
- OpenSSL
- Rust & Cargo >= 1.82.0
- OpenMP
- BLAS
- LAPACK
- Vulkan SDK (if you are using vulkan)

### Build

```bash
npm run build
```

### Packaging

```bash
npm pack
```
