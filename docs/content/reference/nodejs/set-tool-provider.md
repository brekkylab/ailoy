---
title: setToolProvider
description: Build the tool name by calling func, replacing the provider it had.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function setToolProvider(name, func): void;
```

Build the tool `name` by calling `func`, replacing the provider it had. See [`crate::tool`](https://docs.rs/ailoy/0.3.2/ailoy/tool/index.html)
for how `func` is called.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `name` | `string` |
| `func` | (`args`) => `unknown` |

## Returns

`void`
