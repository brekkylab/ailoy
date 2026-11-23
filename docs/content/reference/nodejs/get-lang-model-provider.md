---
title: getLangModelProvider
description: "How the model desc names ('<org>/<name>', or '<org>') is built: a backend's name, 'custom' for a factory, or null when its org has no provider."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function getLangModelProvider(desc): "cli" | "api" | "bedrock" | "openrouter" | "vllm" | "custom" | null;
```

How the model `desc` names (`'<org>/<name>'`, or `'<org>'`) is built: a backend's name,
`'custom'` for a factory, or `null` when its org has no provider.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `desc` | `string` |

## Returns

`"cli"` \| `"api"` \| `"bedrock"` \| `"openrouter"` \| `"vllm"` \| `"custom"` \| `null`
