---
title: PartDelta
description: "A piece of a part: text, function, value, image or null, with the fields of that part as far as they have arrived."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

A piece of a part: `text`, `function`, `value`, `image` or `null`, with the fields of that
part as far as they have arrived.

## Indexable

```ts
[field: string]: any
```

## Properties

| Property | Type |
| ------ | ------ |
| <a id="property-type"></a> `type` | `"function"` \| `"text"` \| `"value"` \| `"image"` \| `"null"` |
| <a id="property-text"></a> `text?` | `string` |
