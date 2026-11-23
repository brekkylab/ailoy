---
title: CustomLangModel
description: A model of your own, as a factory given to setLangModelProvider returns it.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

A model of your own, as a factory given to `setLangModelProvider` returns it. Its methods are
called with `this` as the model. Without `inferStream`, streaming gives `infer`'s output as
one delta.

## Methods

### infer()

```ts
infer(
   messages, 
   tools, 
   options
): 
  | MessageOutput
| Promise<MessageOutput>;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `messages` | [`Message`](/docs/reference/nodejs/message/)[] |
| `tools` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/)[] |
| `options` | [`LangModelOptions`](/docs/reference/nodejs/lang-model-options/) |

#### Returns

  \| [`MessageOutput`](/docs/reference/nodejs/message-output/)
  \| `Promise`\<[`MessageOutput`](/docs/reference/nodejs/message-output/)\>

***

### inferStream()?

```ts
inferStream?(
   messages, 
   tools, 
   options
): AsyncIterable<MessageDeltaOutput>;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `messages` | [`Message`](/docs/reference/nodejs/message/)[] |
| `tools` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/)[] |
| `options` | [`LangModelOptions`](/docs/reference/nodejs/lang-model-options/) |

#### Returns

`AsyncIterable`\<[`MessageDeltaOutput`](/docs/reference/nodejs/message-delta-output/)\>
