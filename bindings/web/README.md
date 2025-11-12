# ailoy-web

JavaScript binding for Ailoy APIs based on WebAssembly, enabling AI agent development directly in web browsers.

For comprehensive documentation and guides, visit our [official documentation](https://brekkylab.github.io/ailoy).

## Install

```bash
# Using npm
npm install ailoy-web
# Using yarn
yarn add ailoy-web
```

## Quickstart

```typescript
import * as ai from "ailoy-web";

(async () => {
  // Create a new local LangModel
  const lm = await ai.LangModel.newLocal("Qwen/Qwen3-0.6B");

  // Initialize an agent
  const agent = new ai.Agent(lm);

  // Run the agent and get responses
  for await (const resp of agent.run("Please give me a short poem about AI")) {
    console.log(resp);
  }
})();
```

## Vite Configurations

When using [Vite](https://vite.dev/) as your build tool, apply these essential configurations in your `vite.config.js` to ensure ailoy-web works correctly:

- **Add wasm plugin**: Add [vite-plugin-wasm](https://www.npmjs.com/package/vite-plugin-wasm) plugin to use the wasm binary
- **Exclude from optimization**: Add `ailoy-web` to `optimizeDeps.exclude` to prevent bundling during development
- **Enable cross-origin isolation**: Set required headers for `SharedArrayBuffer` support (required for WebAssembly threading)
- **Optimized build chunks**: Configure manual chunks to reduce bundle size

```js
// vite.config.js
import wasm from "vite-plugin-wasm";

export default defineConfig({
  // ... other config
  plugins: [
    // ... other plugins
    wasm(),
  ],
  optimizeDeps: {
    exclude: ["ailoy-web"],
  },
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          ailoy: ["ailoy-web"]
        },
      },
    },
  },
  // ... other config
});
```

> Note: Cross-origin isolation headers are required because ailoy-web uses `SharedArrayBuffer` for threading. Learn more about [cross-origin isolation](https://web.dev/articles/coop-coep).


## Building from source

### Prerequisites

Ensure you have the following tools installed:

- **Rust** >= 1.88
- **Node.js** >= LTS version
- **C/C++ compiler** (recommended versions are below)
  - GCC >= 13
  - LLVM Clang >= 17
  - Apple Clang >= 15
  - MSVC >= 19.29
- **Emscripten** >= 4.0.0
- **CMake** >= 3.28.0
- **Git**
- **Docker Engine** (required to build faiss shim)


### Build Process

```bash
# Install Node.js dependencies
npm install

# Ailoy uses some functionalities (tvm.js and faiss) from the shim implemented in the javascript level.
# Build the shim first.
npm run build:shim

# Build WebAssembly module
npm run build:wasm

# Bundle files into ./dist
npm run build:ts

# Run every build at once
npm run build
```

> [!WARNING]
> To build binding, you must change the crate type to **`cdylib`**.
>
> ```toml
> [lib]
> crate-type = ["dylib"] # <- Change this to ["cdylib"]
> ```

### Testing

The project uses Vitest with Playwright for comprehensive testing.

```bash
# Setup API keys for testing (optional - enable specific provider tests)
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
export CLAUDE_API_KEY="<YOUR_CLAUDE_API_KEY>"
export XAI_API_KEY="<YOUR_XAI_API_KEY>"

# Run the test suites
npm run test
```

### Creating Distribution Package

```bash
# Generate npm package
npm pack
```
