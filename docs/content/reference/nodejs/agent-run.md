---
title: AgentRun
description: A turn, iterated with for await.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

A turn, iterated with `for await`.

Nothing runs until the first `next()`, and the turn only goes as far as it has been
iterated: breaking out of the loop, or `return()`, ends it where it stands.

## Constructors

### Constructor

```ts
new AgentRun(): AgentRun;
```

#### Returns

`AgentRun`

## Methods

### next()

```ts
next(): Promise<IteratorResult<
  | MessageOutput
| MessageDeltaOutput, undefined>>;
```

#### Returns

`Promise`\<`IteratorResult`\<
  \| [`MessageOutput`](/docs/reference/nodejs/message-output/)
  \| [`MessageDeltaOutput`](/docs/reference/nodejs/message-delta-output/), `undefined`\>\>

***

### return()

```ts
return(): Promise<IteratorResult<
  | MessageOutput
| MessageDeltaOutput, undefined>>;
```

End the turn where it stands, as breaking out of `for await` does.

#### Returns

`Promise`\<`IteratorResult`\<
  \| [`MessageOutput`](/docs/reference/nodejs/message-output/)
  \| [`MessageDeltaOutput`](/docs/reference/nodejs/message-delta-output/), `undefined`\>\>
