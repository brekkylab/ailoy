---
title: "Agent"
description: "A built agent: runs turns on a query and keeps the conversation."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

An `Agent` is a model, its tools and the conversation so far. Get one from `await AgentBuilder(model)...build()`, or from a spec with `Agent.from_spec`.

Each call to `run()` or `run_stream()` is one turn. The query joins the history, the model answers, the agent runs the tools it calls and hands their results back, and the model goes on until it answers without calling a tool. The history carries over, so each turn sees everything before it. If the model fails before its first message of a turn, the query is taken back out of the history, so the agent can be run again.

An agent runs one turn at a time. While a turn is being iterated, `history` and `start_browser()` raise [`AiloyError`](../errors/) ("this agent is in the middle of a turn"). After `close()` they raise "this agent has been closed".

`Agent` is an async context manager: `async with await builder.build() as agent:` closes it on the way out.

## Example

```python
import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").build() as agent:
        # run(): each whole message, once it's complete
        async for out in agent.run("Name three prime numbers."):
            for part in out["message"]["contents"]:
                if part["type"] == "text":
                    print(part["text"])

        # run_stream(): the model's words as they're written. This turn sees the last one.
        async for delta in agent.run_stream("Now add them up."):
            for piece in delta["delta"]["contents"]:
                print(piece.get("text", ""), end="", flush=True)
        print()

        print(len(agent.history), "messages so far")


asyncio.run(main())
```

Guides: [Running an agent](/docs/guides/running/), [Defining an agent](/docs/guides/agents/).

## Members

| Member | Description |
| --- | --- |
| [`from_spec()`](#from_spec) | An agent from a spec, the dict an `AgentSpec` serializes to, for a caller who holds one rather than building it up. |
| [`history`](#history) | The conversation so far, as a list of messages. |
| [`start_browser()`](#start_browser) | Connect the browser now rather than on the agent's first use. |
| [`run()`](#run) | Run one turn on `query`, a message dict, or a string as the user's text, and iterate what it produces: each complete message, the model's and each tool's. |
| [`run_stream()`](#run_stream) | Run one turn as `run` does, but iterate the model's output as it is generated: deltas for each message the model writes, and each tool's result whole. |
| [`close()`](#close) | Let go of the agent now, and of its console if nothing else holds it. |

## Properties

### `history`

The conversation so far, as a list of messages.

Read without waiting, so it raises while a turn holds the agent rather than making
this property a coroutine.

It holds whole messages, from streamed turns too: the system message, each query, and every message the model and the tools produced.

```python
history: list[Message]
```

**Type** `list[`[`Message`](/docs/reference/python/messages/#message)`]`

## Methods

### `from_spec()`

An agent from a spec, the dict an `AgentSpec` serializes to, for a caller who holds
one rather than building it up. Awaitable.

The spec gives the model, instruction, tools and skills. The rest of what a builder sets comes from the options here.

```python
@staticmethod
async def from_spec(
    spec: AgentSpec,
    *,
    history: list[Message] | None = None,
    console: ConsoleClient | None = None,
    memory: str | None = None
) -> Agent
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `spec` | [`AgentSpec`](/docs/reference/python/specs-and-options/#agentspec) | required | The agent as data. See [`AgentSpec`](../specs-and-options/#agentspec). |
| `history` | `list[`[`Message`](/docs/reference/python/messages/#message)`] \| None` | `None` | Earlier messages to start from, as `AgentBuilder.history()` takes them. |
| `console` | `ConsoleClient \| None` | `None` | A `virtx.ConsoleClient` for the console tools. |
| `memory` | `str \| None` | `None` | The memory store's path in the console, as `AgentBuilder.memory()` takes it. |

**Returns** an awaitable of [`Agent`](/docs/reference/python/agent/): The agent.

### `start_browser()`

Connect the browser now rather than on the agent's first use.

This is when the browser asks the person, so call it while they are at their screen; a
refusal raises. A no-op once connected. Raises during a turn.

```python
async def start_browser() -> None
```

### `run()`

Run one turn on `query`, a message dict, or a string as the user's text, and
iterate what it produces: each complete message, the model's and each tool's.

```python
def run(query: str | Message) -> AgentRun[MessageOutput]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str \| `[`Message`](/docs/reference/python/messages/#message) | required | What to ask: a string, as the user's text, or a whole [`Message`](../messages/#message). |

**Returns** [`AgentRun`](/docs/reference/python/agent-run/)`[`[`MessageOutput`](/docs/reference/python/outputs/#messageoutput)`]`: The turn, which runs as it's iterated. See [`AgentRun`](../agent-run/).

### `run_stream()`

Run one turn as `run` does, but iterate the model's output as it is generated: deltas
for each message the model writes, and each tool's result whole.

```python
def run_stream(query: str | Message) -> AgentRun[MessageDeltaOutput]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str \| `[`Message`](/docs/reference/python/messages/#message) | required | What to ask: a string, as the user's text, or a whole [`Message`](../messages/#message). |

**Returns** [`AgentRun`](/docs/reference/python/agent-run/)`[`[`MessageDeltaOutput`](/docs/reference/python/outputs/#messagedeltaoutput)`]`: The turn, which runs as it's iterated. See [`AgentRun`](../agent-run/).

### `close()`

Let go of the agent now, and of its console if nothing else holds it. Waits for a turn
being iterated to finish. Closing twice is the same as closing once.

```python
async def close() -> None
```
