---
title: "Agent"
description: "A built agent: runs turns on a query and keeps the conversation."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

An `Agent` is a model, its tools and the conversation so far. Get one from `await new AgentBuilder(model)...build()`, or from a spec with `Agent.fromSpec`.

Each call to `run()` or `runStream()` is one turn. The query joins the history, the model answers, the agent runs the tools it calls and hands their results back, and the model goes on until it answers without calling a tool. The history carries over, so each turn sees everything before it. If the model fails before its first message of a turn, the query is taken back out of the history, so the agent can be run again.

An agent runs one turn at a time. While a turn is being iterated, `history` and `startBrowser()` throw an [`AILOY_ERROR`](../errors/) ("this agent is in the middle of a turn"). After `close()` they throw "this agent has been closed".

## Example

```js
const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').build()

  // run(): each whole message, once it's complete
  for await (const out of agent.run('Name three prime numbers.')) {
    for (const part of out.message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }

  // runStream(): the model's words as they're written. This turn sees the last one.
  for await (const out of agent.runStream('Now add them up.')) {
    for (const piece of out.delta.contents) process.stdout.write(piece.text ?? '')
  }
  console.log()

  console.log(agent.history.length, 'messages so far')
  await agent.close()
}

main()
```

Guides: [Running an agent](/docs/guides/running/), [Defining an agent](/docs/guides/agents/).

## Members

| Member | Description |
| --- | --- |
| [`fromSpec()`](#fromspec) | An agent from a spec (the object an `AgentSpec` serializes to). |
| [`history`](#history) | The conversation so far, as a list of messages. |
| [`startBrowser()`](#startbrowser) | Connect the agent's browser now rather than on its first use. |
| [`run()`](#run) | Run one turn on `query`, a message, or a string as the user's text, and iterate what it produces: each complete message, the model's and each tool's. |
| [`runStream()`](#runstream) | Run one turn as `run` does, but iterate the model's output as it is generated: deltas for each message the model writes, and each tool's result whole. |
| [`close()`](#close) | Let go of the agent now, and of its console if nothing else holds it. |

## Properties

### `history`

The conversation so far, as a list of messages.

Read without waiting, so it throws while a turn holds the agent rather than making
this getter a promise.

It holds whole messages, from streamed turns too: the system message, each query, and every message the model and the tools produced.

```ts
get history(): Message[]
```

**Type** [`Message`](/docs/reference/nodejs/messages/#message)`[]`

## Methods

### `fromSpec()`

An agent from a spec (the object an `AgentSpec` serializes to). `options.memory` is the
file to remember into.

The spec gives the model, instruction, tools and skills. The rest of what a builder sets comes from the options here.

```ts
static fromSpec(
  spec: AgentSpec,
  options?: { history?: Message[]; console?: ConsoleClient; memory?: string },
): Promise<Agent>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `spec` | [`AgentSpec`](/docs/reference/nodejs/specs-and-options/#agentspec) | required | The agent as data. See [`AgentSpec`](../specs-and-options/#agentspec). |
| `options` (optional) | `{ history?: `[`Message`](/docs/reference/nodejs/messages/#message)`[]; console?: ConsoleClient; memory?: string }` |  | What the agent starts with besides its spec: its history, console and memory. |
| `options.history` (optional) | [`Message`](/docs/reference/nodejs/messages/#message)`[]` |  | Earlier messages to start from, as `AgentBuilder.history()` takes them. |
| `options.console` (optional) | `ConsoleClient` |  | A `ConsoleClient` from `@brekkylab/virtx` for the console tools. |
| `options.memory` (optional) | `string` |  | The memory store's path in the console, as `AgentBuilder.memory()` takes it. |

**Returns** `Promise<`[`Agent`](/docs/reference/nodejs/agent/)`>`: The agent.

### `startBrowser()`

Connect the agent's browser now rather than on its first use. This is when the browser asks,
so call it while the person is at their screen; a refusal rejects. A no-op once
connected. Throws during a turn.

```ts
startBrowser(): Promise<void>
```

### `run()`

Run one turn on `query`, a message, or a string as the user's text, and iterate what
it produces: each complete message, the model's and each tool's.

```ts
run(query: string | Message): AsyncIterableIterator<MessageOutput>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `string \| `[`Message`](/docs/reference/nodejs/messages/#message) | required | What to ask: a string, as the user's text, or a whole [`Message`](../messages/#message). |

**Returns** `AsyncIterableIterator<`[`MessageOutput`](/docs/reference/nodejs/outputs/#messageoutput)`>`: The turn, which runs as it's iterated. See [`AgentRun`](../agent-run/).

### `runStream()`

Run one turn as `run` does, but iterate the model's output as it is generated: deltas
for each message the model writes, and each tool's result whole.

```ts
runStream(query: string | Message): AsyncIterableIterator<MessageDeltaOutput>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `string \| `[`Message`](/docs/reference/nodejs/messages/#message) | required | What to ask: a string, as the user's text, or a whole [`Message`](../messages/#message). |

**Returns** `AsyncIterableIterator<`[`MessageDeltaOutput`](/docs/reference/nodejs/outputs/#messagedeltaoutput)`>`: The turn, which runs as it's iterated. See [`AgentRun`](../agent-run/).

### `close()`

Let go of the agent now, and of its console if nothing else holds it. Waits for a turn
being iterated to finish. Closing twice is the same as closing once.

```ts
close(): Promise<void>
```
