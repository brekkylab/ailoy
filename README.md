# Ailoy

[![document](https://img.shields.io/badge/document-latest-2ea44f?color=5a9cae)](https://brekkylab.github.io/ailoy/)
[![Python version](https://img.shields.io/pypi/pyversions/ailoy-py)](https://pypi.org/project/ailoy-py/)
[![Node version](https://img.shields.io/node/v/ailoy-node?color=339933)](https://www.npmjs.com/package/ailoy-node)


[![ailoy-py version](https://img.shields.io/pypi/v/ailoy-py?color=blue)](https://pypi.org/project/ailoy-py/)
[![ailoy-node version](<https://img.shields.io/npm/v/ailoy-node?label=npm(node)&color=339933>)](https://www.npmjs.com/package/ailoy-node)
[![ailoy-web version](<https://img.shields.io/npm/v/ailoy-web?label=npm(web)&color=654ff0>)](https://www.npmjs.com/package/ailoy-web)

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/27rx3EJy3P)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Failoy_co)](https://x.com/ailoy_co)


Ailoy is a library for building **AI agents** and **AI applications**.
It is designed to enable AI features effortlessly. It is intended to be used just by importing it.

## Features

- Comprehensive AI agent capabilities
  - Reasoning-based workflows
  - **Tool calling** capabilities, including **MCP** integration
  - Native **RAG(Retrieval-Augmented Generation)** support with built-in vector store
- Run AI models locally or via cloud APIs
  - Easily **switch between open-source models and APIs**
  - **Fully local and private agent support**
- Cross-platform compatibility
  - Runs in almost any environment
  - Supports native-equivalent functionality in **web browsers** using WASM

For more details, please refer to the [documentation](https://brekkylab.github.io/ailoy/).

> [!WARNING]
> This library is in an early stage of development. APIs may change without notice.

> [!TIP]
> We have a [Discord channel](https://discord.gg/27rx3EJy3P)! If you get stuck or have any questions, feel free to join and ask.

## Requirements

Please refer to the [documentation](https://brekkylab.github.io/ailoy/docs/resources/supported-environments).

## Build from source

### Python Build

```bash
cd bindings/python
maturin build --out ./dist
```

For more details, refer to `bindings/python/README.md`.

### Node.js Build

```bash
cd bindings/nodejs
npm run build
```

For more details, refer to `bindings/nodejs/README.md`.

### Javascript Build for web browsers

```bash
cd bindings/web
npm run build
```

For more details, refer to `bindings/web/README.md`.
