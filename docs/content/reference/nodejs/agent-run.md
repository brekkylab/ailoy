---
title: "AgentRun"
description: "One turn of an agent, iterated as it runs."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

`agent.run()` and `agent.runStream()` return an `AgentRun`, the turn before it has started, typed as an `AsyncIterableIterator`. Iterate it with `for await`.

Nothing runs until the first `next()`, and the turn only goes as far as it has been iterated. Breaking out of the loop, or `return()`, ends the turn where it stands.

What it yields depends on where it came from. `run()` yields a [`MessageOutput`](../outputs/#messageoutput) per complete message: the model's, then each tool's result. `runStream()` yields [`MessageDeltaOutput`](../outputs/#messagedeltaoutput)s: pieces of the model's message as it's generated, and each tool's result whole. Errors from the model, a tool's provider or the console reject the `next()` they happen in, which `for await` throws.

## Example

```js
const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').webFetchTool().build()

  const turn = agent.run('Read https://example.com and summarize it.') // nothing has run yet
  for await (const out of turn) {
    console.log(out.message.role, out.finish_reason.type)
    if (out.message.role === 'tool') break // ends the turn here, after the first tool result
  }
  await agent.close()
}

main()
```

Guides: [Running an agent](/docs/guides/running/).

## Members

| Member | Description |
| --- | --- |
| [`next()`](#next) | Run the turn on to its next output. |
| [`return()`](#return) | End the turn where it stands, as breaking out of `for await` does. |

## Methods

### `next()`

Run the turn on to its next output. Done once the turn has ended.

```ts
next(): Promise<IteratorResult<MessageOutput | MessageDeltaOutput, undefined>>
```

**Returns** `Promise<IteratorResult<`[`MessageOutput`](/docs/reference/nodejs/outputs/#messageoutput)` | `[`MessageDeltaOutput`](/docs/reference/nodejs/outputs/#messagedeltaoutput)`, undefined>>`: The next output, or done.

### `return()`

End the turn where it stands, as breaking out of `for await` does.

```ts
return(): Promise<IteratorResult<MessageOutput | MessageDeltaOutput, undefined>>
```

**Returns** `Promise<IteratorResult<`[`MessageOutput`](/docs/reference/nodejs/outputs/#messageoutput)` | `[`MessageDeltaOutput`](/docs/reference/nodejs/outputs/#messagedeltaoutput)`, undefined>>`: Done.
