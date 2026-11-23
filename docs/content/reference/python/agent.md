---
title: Agent
description: "Python reference for Agent: runs turns and holds the history."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `ailoy-py`.*

## `from_spec`

```python
@staticmethod
async from_spec(
    spec: AgentSpec,
    *,
    history: list[Message] | None = None,
    console: ConsoleClient | None = None,
    memory: str | None = None
) -> Agent
```

An agent from a spec, the dict an `AgentSpec` serializes to, for a caller who holds
one rather than building it up. Awaitable.

## `history`

```python
history: list[Message]
```

The conversation so far, as a list of messages.

Read without waiting, so it raises while a turn holds the agent rather than making
this property a coroutine.

## `start_browser`

```python
async start_browser() -> None
```

Connect the browser now rather than on the agent's first use.

This is when the browser asks the person, so call it while they are at their screen; a
refusal raises. A no-op once connected. Raises during a turn.

## `run`

```python
run(query: str | Message) -> AgentRun[MessageOutput]
```

Run one turn on `query`, a message dict, or a string as the user's text, and
iterate what it produces: each complete message, the model's and each tool's.

## `run_stream`

```python
run_stream(query: str | Message) -> AgentRun[MessageDeltaOutput]
```

Run one turn as `run` does, but iterate the model's output as it is generated: deltas
for each message the model writes, and each tool's result whole.

## `close`

```python
async close() -> None
```

Let go of the agent now, and of its console if nothing else holds it. Waits for a turn
being iterated to finish. Closing twice is the same as closing once.
