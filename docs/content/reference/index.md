---
title: API reference
description: "The Ailoy SDK reference for Python, Node.js and Rust, generated from the v0.3.2 release."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

Every class, function and type the Ailoy SDK exports, per language. It's generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release, so it describes the published packages. The guides follow `main`, which can be ahead of the release.

| Language | Package | Reference |
| --- | --- | --- |
| Python | `ailoy-py` 0.3.2 | [Python API reference](/docs/reference/python/) |
| Node.js | `@brekkylab/ailoy` 0.3.2 | [Node.js API reference](/docs/reference/nodejs/) |
| Rust | `ailoy` 0.3.2 | [docs.rs/ailoy/0.3.2](https://docs.rs/ailoy/0.3.2/ailoy/) |

Where to start, in any language:

- `AgentBuilder`: sets up an agent and builds it. [Python](/docs/reference/python/agent-builder/), [Node.js](/docs/reference/nodejs/agent-builder/), [Rust](https://docs.rs/ailoy/0.3.2/ailoy/agent/struct.AgentBuilder.html)
- `Agent`: runs turns and holds the history. [Python](/docs/reference/python/agent/), [Node.js](/docs/reference/nodejs/agent/), [Rust](https://docs.rs/ailoy/0.3.2/ailoy/agent/struct.Agent.html)
- `LangModel`: a model run without an agent. [Python](/docs/reference/python/lang-model/), [Node.js](/docs/reference/nodejs/lang-model/)
