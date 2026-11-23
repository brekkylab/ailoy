---
title: registerMcpStreamableHttp
description: Connect to the streamable-HTTP MCP server at url and register each of its tools as {prefix}__{name}, settling with their descriptions.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function registerMcpStreamableHttp(prefix, url): Promise<ToolDesc[]>;
```

Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
`{prefix}__{name}`, settling with their descriptions.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `prefix` | `string` |
| `url` | `string` |

## Returns

`Promise`\<[`ToolDesc`](/docs/reference/nodejs/tool-desc/)[]\>
