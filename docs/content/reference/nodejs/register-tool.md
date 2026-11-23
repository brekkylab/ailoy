---
title: registerTool
description: Build the tool desc describes by calling func, and hand desc back.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

```ts
function registerTool(desc, func): ToolDesc;
```

Build the tool `desc` describes by calling `func`, and hand `desc` back.

`desc.name` is what the model calls and what the provider is set under.

## Parameters

| Parameter | Type |
| ------ | ------ |
| `desc` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/) |
| `func` | (`args`) => `unknown` |

## Returns

[`ToolDesc`](/docs/reference/nodejs/tool-desc/)
