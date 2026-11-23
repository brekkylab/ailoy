---
title: LangModel
description: A model built from its desc, as an Agent builds one, run on a conversation directly.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

A model built from its desc, as an `Agent` builds one, run on a conversation directly.

## Constructors

### Constructor

```ts
new LangModel(): LangModel;
```

#### Returns

`LangModel`

## Methods

### create()

```ts
static create(desc): Promise<LangModel>;
```

The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for
`<org>`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `desc` | `string` |

#### Returns

`Promise`\<`LangModel`\>

***

### infer()

```ts
infer(
   messages, 
   tools?, 
   options?
): Promise<MessageOutput>;
```

Run the model on `messages` to completion, settling with the whole output.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `messages` | [`Message`](/docs/reference/nodejs/message/)[] |
| `tools?` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/)[] |
| `options?` | [`LangModelOptions`](/docs/reference/nodejs/lang-model-options/) |

#### Returns

`Promise`\<[`MessageOutput`](/docs/reference/nodejs/message-output/)\>

***

### inferStream()

```ts
inferStream(
   messages, 
   tools?, 
   options?
): AsyncIterableIterator<MessageDeltaOutput>;
```

Run the model on `messages` and iterate its output as deltas while it is generated.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `messages` | [`Message`](/docs/reference/nodejs/message/)[] |
| `tools?` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/)[] |
| `options?` | [`LangModelOptions`](/docs/reference/nodejs/lang-model-options/) |

#### Returns

`AsyncIterableIterator`\<[`MessageDeltaOutput`](/docs/reference/nodejs/message-delta-output/)\>
