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
| `depth` | `0` for this agent, higher for messages from a sub-agent. |
| `source_agent` | The name of the agent that wrote the message. |

`agent.run_stream` (`runStream` in Node.js) yields `MessageDeltaOutput`s instead. Each `delta` holds a piece of a message, such as a few tokens of text or thinking. Concatenated, the deltas between two `finish_reason`s make up the same message `run` would yield. In Rust, the `Delta` trait's `accumulate` and `finish` do this for you.

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
