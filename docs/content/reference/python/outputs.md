---
title: "Outputs"
description: "What a turn or a model call gives back: whole messages, streamed deltas, finish reasons and token usage."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

`run()` and `infer()` give [`MessageOutput`](#messageoutput)s: a whole message, why it ended, and the tokens it took. `run_stream` and `infer_stream` give [`MessageDeltaOutput`](#messagedeltaoutput)s: pieces of the model's message as it's generated. The one with a `finish_reason` marks the end of a message. Tool results come whole, one output each, and report no usage.

To rebuild a message from its deltas, append each `delta`'s text and parts in order until a `finish_reason` arrives. The agent does this itself, so `history` holds whole messages either way.

## Example

```python
import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").build() as agent:
        # Whole messages: why each ended, and what it cost
        async for out in agent.run("Explain DNS in two sentences."):
            reason = out["finish_reason"]
            usage = out.get("usage")
            if usage:
                print("tokens in/out:", usage["input_tokens"], usage["output_tokens"])
            if reason["type"] == "refusal":
                print("refused:", reason.get("reason"))

        # Deltas: put the text back together until a finish_reason ends the message
        text = ""
        async for delta in agent.run_stream("Now explain it to a child."):
            for piece in delta["delta"]["contents"]:
                text += piece.get("text", "")
            done = delta["finish_reason"]
            if done:
                print(text, f"({done['type']})")
                text = ""


asyncio.run(main())
```

Guides: [Running an agent](/docs/guides/running/), [Message format](/docs/guides/message-format/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`MessageOutput`](#messageoutput) | type | What `Agent.run` yields: one complete message. |
| [`MessageDeltaOutput`](#messagedeltaoutput) | type | What `Agent.run_stream` yields. |
| [`MessageDelta`](#messagedelta) | type | A piece of a message, with the fields that have arrived in it. |
| [`PartDelta`](#partdelta) | type | A piece of a part: `text`, `function`, `value`, `image` or `null`, with the fields of that part as far as they have arrived. |
| [`FinishReason`](#finishreason) | type | `{"type": "stop"}`, `"length"`, `"tool_call"`, or `"refusal"` with a `reason`. |
| [`TokenUsage`](#tokenusage) | type | `input_tokens` is every token of input the call processed, cached or not; `cache_creation_input_tokens` and `cache_read_input_tokens` are the parts of it written to and read from a prompt cache. |

## `MessageOutput`

What `Agent.run` yields: one complete message.

What `Agent.run` yields and `LangModel.infer` returns.

```python
class MessageOutput(TypedDict):
    message: Message
    finish_reason: FinishReason
    usage: NotRequired[TokenUsage]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `message` | [`Message`](/docs/reference/python/messages/#message) | yes | The message. |
| `finish_reason` | [`FinishReason`](/docs/reference/python/outputs/#finishreason) | yes | Why it ended. |
| `usage` | [`TokenUsage`](/docs/reference/python/outputs/#tokenusage) | no | The tokens the model call took. Only a model's messages have it; a tool result doesn't. |

## `MessageDeltaOutput`

What `Agent.run_stream` yields. A `finish_reason` marks the end of a message.

```python
class MessageDeltaOutput(TypedDict):
    delta: MessageDelta
    finish_reason: FinishReason | None
    usage: TokenUsage | None
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `delta` | [`MessageDelta`](/docs/reference/python/outputs/#messagedelta) | yes | The piece of the message. |
| `finish_reason` | [`FinishReason`](/docs/reference/python/outputs/#finishreason)` \| None` | yes | Set on the last piece of a message: why it ended. |
| `usage` | [`TokenUsage`](/docs/reference/python/outputs/#tokenusage)` \| None` | yes | The tokens the model call took, when the backend reports them. |

## `MessageDelta`

A piece of a message, with the fields that have arrived in it.

```python
class MessageDelta(TypedDict):
    role: NotRequired[Role]
    contents: list[PartDelta]
    id: NotRequired[str]
    thinking: NotRequired[str]
    tool_calls: list[PartDelta]
    signature: NotRequired[str]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `role` | [`Role`](/docs/reference/python/messages/#role) | no | Who is writing, on the first piece. |
| `contents` | `list[`[`PartDelta`](/docs/reference/python/outputs/#partdelta)`]` | yes | Pieces of the message's parts. |
| `id` | `str` | no | The message's id, for a tool result. |
| `thinking` | `str` | no | A piece of the model's thinking. |
| `tool_calls` | `list[`[`PartDelta`](/docs/reference/python/outputs/#partdelta)`]` | yes | Pieces of the tool calls. |
| `signature` | `str` | no | The thinking's signature, for the APIs that send one. |

## `PartDelta`

A piece of a part: `text`, `function`, `value`, `image` or `null`, with the
fields of that part as far as they have arrived.

```python
class PartDelta(TypedDict):
    type: Literal['text', 'function', 'value', 'image', 'null']
    text: NotRequired[str]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['text', 'function', 'value', 'image', 'null']` | yes | Which kind of part it's a piece of. |
| `text` | `str` | no | A piece of the text, for a `text` part. |

## `FinishReason`

`{"type": "stop"}`, `"length"`, `"tool_call"`, or `"refusal"` with a
`reason`.

`stop` is a finished answer, `length` an answer the output limit cut short, `tool_call` an answer that calls tools, and `refusal` one the model refused, with its `reason`.

```python
class FinishReason(TypedDict):
    type: Literal['stop', 'length', 'tool_call', 'refusal']
    reason: NotRequired[str]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['stop', 'length', 'tool_call', 'refusal']` | yes | Why the message ended. |
| `reason` | `str` | no | For `refusal`, why. |

## `TokenUsage`

`input_tokens` is every token of input the call processed, cached or not;
`cache_creation_input_tokens` and `cache_read_input_tokens` are the parts of it
written to and read from a prompt cache.

```python
class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: NotRequired[int]
    cache_read_input_tokens: NotRequired[int]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `input_tokens` | `int` | yes | Every token of input the call processed, cached or not. |
| `output_tokens` | `int` | yes | The tokens the model wrote. |
| `cache_creation_input_tokens` | `int` | no | The part of `input_tokens` written to a prompt cache. |
| `cache_read_input_tokens` | `int` | no | The part of `input_tokens` read from a prompt cache. |
