---
title: unregisterMcp
description: Drop the tools registered under prefix and end that server's session.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function unregisterMcp(prefix): number;
```

Drop the tools registered under `prefix` and end that server's session. Answers with how
many were removed.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `prefix` | `string` |

## Returns

`number`
