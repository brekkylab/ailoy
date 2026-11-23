---
title: setLangModelProvider
description: "Build the models of org ('anthropic', 'openai', 'google', 'deepseek-ai', …) through provider: a backend, or a factory, as the module says."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function setLangModelProvider(org, provider): void;
```

Build the models of `org` (`'anthropic'`, `'openai'`, `'google'`, `'deepseek-ai'`, …)
through `provider`: a backend, or a factory, as the module says. Replaces the provider the
org had.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `org` | `string` |
| `provider` | \| `"cli"` \| `"api"` \| `"bedrock"` \| `"openrouter"` \| `"vllm"` \| ((`org`, `name`) => \| [`LangModel`](/docs/reference/nodejs/lang-model/) \| [`CustomLangModel`](/docs/reference/nodejs/custom-lang-model/)) |

## Returns

`void`
