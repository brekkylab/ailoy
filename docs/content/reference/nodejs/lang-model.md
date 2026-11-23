---
title: "LangModel"
description: "A model run directly on a conversation, without an agent."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

A `LangModel` is a model on its own: no history, no turn loop, and no tools run. `create(desc)` builds it through the provider set for the desc's org, the way an agent builds its model, and `infer()` runs it once on the messages you pass.

When the model asks for a tool, the request comes back in the message's `tool_calls`. Running the tool, and sending its result back in the next call, is up to you. Use a `LangModel` for one-off calls, or as what a factory given to [`setLangModelProvider`](../model-providers/) returns.

## Example

```js
const { LangModel } = require('@brekkylab/ailoy')

async function main() {
  const model = await LangModel.create('anthropic/claude-sonnet-5-5')
  const out = await model.infer(
    [
      { role: 'system', contents: [{ type: 'text', text: 'Reply in French.' }] },
      { role: 'user', contents: [{ type: 'text', text: 'Good morning!' }] },
    ],
    [],
    { thinking_effort: 'low' },
  )
  for (const part of out.message.contents) {
    if (part.type === 'text') console.log(part.text)
  }
}

main()
```

Guides: [Model providers](/docs/guides/model-providers/).

## Members

| Member | Description |
| --- | --- |
| [`create()`](#create) | The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for `<org>`. |
| [`infer()`](#infer) | Run the model on `messages` to completion, settling with the whole output. |
| [`inferStream()`](#inferstream) | Run the model on `messages` and iterate its output as deltas while it is generated. |

## Methods

### `create()`

The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for
`<org>`.

```ts
static create(desc: string): Promise<LangModel>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | `string` | required | The model's desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI's default model. See [Model providers](../model-providers/). |

**Returns** `Promise<`[`LangModel`](/docs/reference/nodejs/lang-model/)`>`: The model.

**Throws** `AILOY_ERROR` when the org has no provider, or the backend can't build the model.

### `infer()`

Run the model on `messages` to completion, settling with the whole output.

```ts
infer(
  messages: Message[],
  tools?: ToolDesc[],
  options?: LangModelOptions,
): Promise<MessageOutput>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | [`Message`](/docs/reference/nodejs/messages/#message)`[]` | required | The conversation to run the model on, oldest first. |
| `tools` (optional) | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]` |  | The tools the model may call, by their descs. The model only asks for them: nothing runs them. |
| `options` (optional) | [`LangModelOptions`](/docs/reference/nodejs/specs-and-options/#langmodeloptions) |  | Thinking effort and output schema for this call. |

**Returns** `Promise<`[`MessageOutput`](/docs/reference/nodejs/outputs/#messageoutput)`>`: The model's whole message, why it ended, and the tokens it took.

### `inferStream()`

Run the model on `messages` and iterate its output as deltas while it is generated.

```ts
inferStream(
  messages: Message[],
  tools?: ToolDesc[],
  options?: LangModelOptions,
): AsyncIterableIterator<MessageDeltaOutput>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | [`Message`](/docs/reference/nodejs/messages/#message)`[]` | required | The conversation to run the model on, oldest first. |
| `tools` (optional) | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]` |  | The tools the model may call, by their descs. The model only asks for them: nothing runs them. |
| `options` (optional) | [`LangModelOptions`](/docs/reference/nodejs/specs-and-options/#langmodeloptions) |  | Thinking effort and output schema for this call. |

**Returns** `AsyncIterableIterator<`[`MessageDeltaOutput`](/docs/reference/nodejs/outputs/#messagedeltaoutput)`>`: The model's message as deltas.
