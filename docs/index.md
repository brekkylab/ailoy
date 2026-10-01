---
layout: home

hero:
  name: Ailoy
  text: 
  tagline: AI agent builder with a VM at its heart.
  actions:
    - theme: brand
      text: Get started
      link: /guide/quick-start
    - theme: alt
      text: Examples
      link: /guide/examples
    - theme: alt
      text: View on GitHub
      link: https://github.com/brekkylab/ailoy

---

![Ailoy demo](./images/ailoy-sam3.gif)

## Install

::: code-group

```sh [Python]
pip install ailoy-py
```

```sh [Node.js]
npm install ailoy-node
```

```sh [Rust]
cargo add ailoy
```

:::

Most agent frameworks focus on connecting an LLM to predefined tools.
Ailoy does that too — and also lets the agent operate in a general-purpose computing environment.
See the [quick start](/guide/quick-start) to run your first agent.

::: warning
Ailoy is under active development, and its API may change between versions.
:::
