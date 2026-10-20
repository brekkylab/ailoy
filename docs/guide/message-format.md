# Message Format

Everything that goes into and comes out of an agent is a **message**. Ailoy uses one format for every provider, and converts it to the provider's own format when it calls the model.

A message has a **role** and a list of **parts**:

```json
{
  "role": "user",
  "contents": [
    { "type": "text", "text": "Can you explain how photosynthesis works?" }
  ]
}
```

In Python and Node.js a message is a plain dict or object of this shape. In Rust it is `ailoy::message::Message`.

## Role

| Role | Who writes it |
| --- | --- |
| `system` | Instructions that set the model's behaviour. Always the first message. |
| `user` | The end user. |
| `assistant` | The model. |
| `tool` | A tool, with the result of a call the model asked for. |

You don't usually write the system message yourself. The agent builds it from `.instruction(...)` (and any skills) and puts it at the front of the history, unless the history you give it already has one.

## Fields

| Field | Description |
| --- | --- |
| `role` | One of the roles above. |
| `contents` | What was said, as a list of parts. |
| `thinking` | The model's reasoning trace, kept apart from `contents`. Only on assistant messages from reasoning models. |
| `tool_calls` | The tools the model wants to call, as `function` parts. Only on assistant messages. |
| `id` | On a `tool` message, the id of the call it answers. |
| `signature` | A provider-issued signature for `thinking`, sent back as is on the next turn. |

Reasoning and tool calls are stored in their own fields rather than in `contents`, so the answer the user sees is never mixed with them.

## Part

A part is one unit of content. A message made of text, an image and more text is three parts.

| `type` | Shape | Used for |
| --- | --- | --- |
| `text` | `{ "text": "..." }` | Plain text. |
| `image` | `{ "image": { "type": "url", "url": "..." } }` or `{ "image": { "type": "embedded", "mime_type": "image/png", "data": ... } }` | An image, by URL or as bytes. |
| `function` | `{ "id": "...", "function": { "name": "...", "arguments": { ... } } }` | A tool call, inside `tool_calls`. |
| `value` | `{ "value": ... }` | Any JSON value, such as a tool result. |

::: tip
Not every provider accepts image URLs (Gemini doesn't). Embedded bytes work everywhere that takes images.
:::

## Sending a message

`agent.run` takes either a string, which becomes one user text part, or a full message. Pass a message when you need more than text, such as an image:

::: code-group

```python [Python]
from pathlib import Path

query = {
    "role": "user",
    "contents": [
        {"type": "image", "image": {"type": "embedded", "mime_type": "image/jpeg", "data": Path("dog.jpg").read_bytes()}},
        {"type": "text", "text": "What do you see in this image?"},
    ],
}
async for output in agent.run(query):
    ...
```

```js [Node.js]
const fs = require('fs')

const query = {
  role: 'user',
  contents: [
    { type: 'image', image: { type: 'embedded', mime_type: 'image/jpeg', data: fs.readFileSync('dog.jpg') } },
    { type: 'text', text: 'What do you see in this image?' },
  ],
}
for await (const output of agent.run(query)) {
  // ...
}
```

```rust [Rust]
use ailoy::message::{Message, Part, Role};

let query = Message::new(Role::User).with_contents([
    Part::image_embedded("image/jpeg", std::fs::read("dog.jpg")?.into())?,
    Part::text("What do you see in this image?"),
]);
let mut stream = agent.run(query);
```

:::

## Tool calls and results

When the model decides to call a tool, its assistant message carries the call in `tool_calls`:

```json
{
  "role": "assistant",
  "contents": [{ "type": "text", "text": "Let me check the weather." }],
  "tool_calls": [
    {
      "type": "function",
      "id": "call_01HZX2",
      "function": { "name": "get_weather", "arguments": { "city": "Seoul" } }
    }
  ]
}
```

The agent runs the tool and adds a `tool` message whose `id` matches the call:

```json
{
  "role": "tool",
  "id": "call_01HZX2",
  "contents": [{ "type": "value", "value": 12.3 }]
}
```

If the tool fails, the error still comes back as a `tool` message, so the model can see it and recover. The agent then calls the model again, and repeats until the model answers without calling a tool.

## What `run` yields

`agent.run` yields one `MessageOutput` for each message of the turn: the assistant's tool calls, each tool result, and the final answer.

| Field | Description |
| --- | --- |
| `message` | The complete message. |
| `finish_reason` | Why the model stopped: `stop`, `length`, `tool_call` or `refusal` (with a `reason`). |
| `usage` | Input and output token counts, when the provider reports them. |

`agent.run_stream` (`runStream` in Node.js) yields `MessageDeltaOutput`s instead. Each `delta` holds a piece of a message, such as a few tokens of text or thinking. Concatenated, the deltas between two `finish_reason`s make up the same message `run` would yield. In Rust, the `Delta` trait's `accumulate` and `finish` do this for you, and `into_messages` turns a whole `run_stream` into what `run` yields.

### A tool message that streams

A tool's result streams the way an assistant message does. While the tool runs, `run_stream` yields each piece it produces as a delta with role `tool` and the call's `id`, and a delta whose `finish_reason` is `stop` ends the result:

```json
{ "delta": { "role": "tool", "id": "call_01HZX2", "contents": [{ "type": "text", "text": "looking up Seoul, " }] }, "finish_reason": null }
{ "delta": { "role": "tool", "id": "call_01HZX2", "contents": [{ "type": "text", "text": "done" }] }, "finish_reason": null }
{ "delta": { "role": "tool", "id": "call_01HZX2", "contents": [{ "type": "value", "value": { "celsius": 12.3 } }] }, "finish_reason": { "type": "stop" } }
```

Adjacent text joins into one part and a value or an image is a part of its own, so these three make a result of two parts: the text `looking up Seoul, done` and the value. A tool that returns once arrives as one delta that carries its `finish_reason`.

The calls the model asks for together run at the same time, so their deltas interleave. Put a tool result together by its `id`, not by the order the deltas arrive in; `into_messages` does.

What streams is the result itself: the message its deltas make up is the one `run` yields, the one the history keeps, and the one the model reads next. Each part of a tool result is cut to 30,000 characters, keeping its start and end. Each delta is cut the same way as it passes, so a program reading the deltas may see more of a long result than the history keeps.

A tool written as an async generator streams its result: each value it yields is a piece, a string as text and anything else as a value. Breaking out of the turn closes the generator, so its `finally` runs.

::: code-group

```python [Python]
import asyncio

import ailoy

WEATHER = {
    "name": "weather",
    "description": "The weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


async def weather(city: str):
    yield f"looking up {city}, "
    await asyncio.sleep(1)  # the lookup
    yield "done"
    yield {"celsius": 12.3}


desc = ailoy.register_tool(WEATHER, weather)
```

```js [Node.js]
const WEATHER = {
  name: 'weather',
  description: 'The weather in a city.',
  parameters: { type: 'object', properties: { city: { type: 'string' } } },
}

const desc = registerTool(WEATHER, async function* ({ city }) {
  yield `looking up ${city}, `
  await new Promise((resolve) => setTimeout(resolve, 1000)) // the lookup
  yield 'done'
  yield { celsius: 12.3 }
})
```

```rust [Rust]
use ailoy::{datatype::Value, message::{Message, Part, Role}, tool_func};

// Each message the stream yields is a piece of the result.
let weather = tool_func!(stream |args: Value, id: String| -> Message {
    let piece = move |part: Part| Message::new(Role::Tool).with_contents([part]).with_id(id.clone());
    futures::stream::iter([
        piece(Part::text("looking up Seoul, ")),
        piece(Part::text("done")),
        piece(Part::value(ailoy::to_value!({"celsius": 12.3}))),
    ])
});
```

:::

## History

An agent keeps the conversation: every query, assistant message and tool result is added to its history, so the next `run` continues where the last one left off. Read it with `agent.history`.

To resume an earlier conversation, pass the saved messages to the builder:

::: code-group

```python [Python]
agent = await AgentBuilder("anthropic/claude-haiku-4-5").history(saved_messages).build()
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-4-5').history(savedMessages).build()
```

```rust [Rust]
let agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
    .history(saved_messages)
    .build()
    .await?;
```

:::

Messages are plain JSON, so you can store them wherever you like and load them back unchanged.
