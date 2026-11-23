---
title: registerMcpStdio
description: Start the MCP server command args… on this host and register each of its tools as {prefix}__{name}, settling with their descriptions.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function registerMcpStdio(
   prefix, 
   command, 
   args?
): Promise<ToolDesc[]>;
```

Start the MCP server `command args…` on this host and register each of its tools as
`{prefix}__{name}`, settling with their descriptions.

The server runs outside any console, with this process's access, register only servers
you trust.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `prefix` | `string` |
| `command` | `string` |
| `args?` | `string`[] \| `null` |

## Returns

`Promise`\<[`ToolDesc`](/docs/reference/nodejs/tool-desc/)[]\>
