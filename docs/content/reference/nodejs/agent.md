---
title: Agent
description: Node.js reference for Agent.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

## Constructors

### Constructor

```ts
new Agent(): Agent;
```

#### Returns

`Agent`

## Accessors

### history

#### Get Signature

```ts
get history(): Message[];
```

The conversation so far, as a list of messages.

Read without waiting, so it throws while a turn holds the agent rather than making
this getter a promise.

##### Returns

[`Message`](/docs/reference/nodejs/message/)[]

## Methods

### fromSpec()

```ts
static fromSpec(spec, options?): Promise<Agent>;
```

An agent from a spec (the object an `AgentSpec` serializes to). `options.memory` is the
file to remember into.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `spec` | [`AgentSpec`](/docs/reference/nodejs/agent-spec/) |
| `options?` | \{ `history?`: [`Message`](/docs/reference/nodejs/message/)[]; `console?`: `ConsoleClient`; `memory?`: `string`; \} |
| `options.history?` | [`Message`](/docs/reference/nodejs/message/)[] |
| `options.console?` | `ConsoleClient` |
| `options.memory?` | `string` |

#### Returns

`Promise`\<`Agent`\>

***

### startBrowser()

```ts
startBrowser(): Promise<void>;
```

Connect the agent's browser now rather than on its first use. This is when the browser asks,
so call it while the person is at their screen; a refusal rejects. A no-op once
connected. Throws during a turn.

#### Returns

`Promise`\<`void`\>

***

### run()

```ts
run(query): AsyncIterableIterator<MessageOutput>;
```

Run one turn on `query`, a message, or a string as the user's text, and iterate what
it produces: each complete message, the model's and each tool's.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `query` | `string` \| [`Message`](/docs/reference/nodejs/message/) |

#### Returns

`AsyncIterableIterator`\<[`MessageOutput`](/docs/reference/nodejs/message-output/)\>

***

### runStream()

```ts
runStream(query): AsyncIterableIterator<MessageDeltaOutput>;
```

Run one turn as `run` does, but iterate the model's output as it is generated: deltas
for each message the model writes, and each tool's result whole.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `query` | `string` \| [`Message`](/docs/reference/nodejs/message/) |

#### Returns

`AsyncIterableIterator`\<[`MessageDeltaOutput`](/docs/reference/nodejs/message-delta-output/)\>

***

### close()

```ts
close(): Promise<void>;
```

Let go of the agent now, and of its console if nothing else holds it. Waits for a turn
being iterated to finish. Closing twice is the same as closing once.

#### Returns

`Promise`\<`void`\>
