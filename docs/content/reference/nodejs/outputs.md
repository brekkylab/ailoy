---
title: "Outputs"
description: "What a turn or a model call gives back: whole messages, streamed deltas, finish reasons and token usage."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

`run()` and `infer()` give [`MessageOutput`](#messageoutput)s: a whole message, why it ended, and the tokens it took. `runStream` and `inferStream` give [`MessageDeltaOutput`](#messagedeltaoutput)s: pieces of the model's message as it's generated. The one with a `finish_reason` marks the end of a message. Tool results come whole, one output each, and report no usage.

To rebuild a message from its deltas, append each `delta`'s text and parts in order until a `finish_reason` arrives. The agent does this itself, so `history` holds whole messages either way.

## Example

```js
const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').build()

  // Whole messages: why each ended, and what it cost
  for await (const out of agent.run('Explain DNS in two sentences.')) {
    if (out.usage) console.log('tokens in/out:', out.usage.input_tokens, out.usage.output_tokens)
    if (out.finish_reason.type === 'refusal') console.log('refused:', out.finish_reason.reason)
  }

  // Deltas: put the text back together until a finish_reason ends the message
  let text = ''
  for await (const out of agent.runStream('Now explain it to a child.')) {
    for (const piece of out.delta.contents) text += piece.text ?? ''
    if (out.finish_reason) {
      console.log(text, `(${out.finish_reason.type})`)
      text = ''
    }
  }
  await agent.close()
}

main()
```

Guides: [Running an agent](/docs/guides/running/), [Message format](/docs/guides/message-format/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`MessageOutput`](#messageoutput) | type | What `Agent.run` yields: one complete message. |
| [`MessageDeltaOutput`](#messagedeltaoutput) | type | What `Agent.runStream` yields. |
| [`MessageDelta`](#messagedelta) | type | A piece of a message, with the fields that have arrived in it. |
| [`PartDelta`](#partdelta) | type | A piece of a part: `text`, `function`, `value`, `image` or `null`, with the fields of that part as far as they have arrived. |
| [`FinishReason`](#finishreason) | type | `{ type: 'stop' }`, `'length'`, `'tool_call'`, or `'refusal'` with a `reason`. |
| [`TokenUsage`](#tokenusage) | type | The tokens a model call took. |

## `MessageOutput`

What `Agent.run` yields: one complete message.

What `agent.run` yields and `LangModel.infer` returns.

```ts
interface MessageOutput {
  message: Message;
  finish_reason: FinishReason;
  usage?: TokenUsage;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `message` | [`Message`](/docs/reference/nodejs/messages/#message) | yes | The message. |
| `finish_reason` | [`FinishReason`](/docs/reference/nodejs/outputs/#finishreason) | yes | Why it ended. |
| `usage` | [`TokenUsage`](/docs/reference/nodejs/outputs/#tokenusage) | no | The tokens the model call took. Only a model's messages have it; a tool result doesn't. |

## `MessageDeltaOutput`

What `Agent.runStream` yields. A `finish_reason` marks the end of a message.

```ts
interface MessageDeltaOutput {
  delta: MessageDelta;
  finish_reason: FinishReason | null;
  usage: TokenUsage | null;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `delta` | [`MessageDelta`](/docs/reference/nodejs/outputs/#messagedelta) | yes | The piece of the message. |
| `finish_reason` | [`FinishReason`](/docs/reference/nodejs/outputs/#finishreason)` \| null` | yes | Set on the last piece of a message: why it ended. |
| `usage` | [`TokenUsage`](/docs/reference/nodejs/outputs/#tokenusage)` \| null` | yes | The tokens the model call took, when the backend reports them. |

## `MessageDelta`

A piece of a message, with the fields that have arrived in it.

```ts
interface MessageDelta {
  role?: Role;
  contents: PartDelta[];
  id?: string;
  thinking?: string;
  tool_calls: PartDelta[];
  signature?: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `role` | [`Role`](/docs/reference/nodejs/messages/#role) | no | Who is writing, on the first piece. |
| `contents` | [`PartDelta`](/docs/reference/nodejs/outputs/#partdelta)`[]` | yes | Pieces of the message's parts. |
| `id` | `string` | no | The message's id, for a tool result. |
| `thinking` | `string` | no | A piece of the model's thinking. |
| `tool_calls` | [`PartDelta`](/docs/reference/nodejs/outputs/#partdelta)`[]` | yes | Pieces of the tool calls. |
| `signature` | `string` | no | The thinking's signature, for the APIs that send one. |

## `PartDelta`

A piece of a part: `text`, `function`, `value`, `image` or `null`, with the fields of that
part as far as they have arrived.

```ts
interface PartDelta {
  type: "function" | "text" | "value" | "image" | "null";
  text?: string;
  [field: string]: any;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"function" \| "text" \| "value" \| "image" \| "null"` | yes | Which kind of part it's a piece of. |
| `text` | `string` | no | A piece of the text, for a `text` part. |

## `FinishReason`

`{ type: 'stop' }`, `'length'`, `'tool_call'`, or `'refusal'` with a `reason`.

`stop` is a finished answer, `length` an answer the output limit cut short, `tool_call` an answer that calls tools, and `refusal` one the model refused, with its `reason`.

```ts
interface FinishReason {
  type: "stop" | "length" | "tool_call" | "refusal";
  reason?: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"stop" \| "length" \| "tool_call" \| "refusal"` | yes | Why the message ended. |
| `reason` | `string` | no | For `refusal`, why. |

## `TokenUsage`

The tokens a model call took.

```ts
interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `input_tokens` | `number` | yes | Every token of input the call processed, cached or not. |
| `output_tokens` | `number` | yes | The tokens the model wrote. |
| `cache_creation_input_tokens` | `number` | no | The part of `input_tokens` written to a prompt cache. |
| `cache_read_input_tokens` | `number` | no | The part of `input_tokens` read from a prompt cache. |
