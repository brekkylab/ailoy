---
title: "Errors"
description: "How ailoy reports failures: what is raised, and where."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

ailoy has no error classes of its own. Every failure is an `Error` whose `code` says what kind it is:

| `code` | Meaning |
| --- | --- |
| `AILOY_ERROR` | Everything ailoy itself reports: a model whose org has no provider, a tool with no provider, an API error, an agent used during a turn or after `close()`. |
| `INVALID_ARG` | An argument ailoy can't take: a builder used after `build()`, an unknown `userBrowser`, a negative token count. |
| `TIMED_OUT`, `NOT_FOUND`, `NOT_EXECUTABLE`, `IS_A_DIRECTORY`, `IO_FAILED`, `BOOT_FAILED`, ... | A console refusal, by the name `@brekkylab/virtx` gives its number, the same as from `ConsoleClient.exec`. |
| `CONSOLE_REFUSED` | A console refusal whose number has no name, with the number in the message. |
| `CONSOLE_BROKEN` | A console whose channel is gone. |

```ts
try {
  for await (const out of agent.run('hello')) console.log(out.message);
} catch (e) {
  if (e.code === 'AILOY_ERROR') console.error('ailoy:', e.message);
  else throw e;
}
```

A turn's errors reject the iterator's `next()`, which `for await` throws. A tool function's own throw isn't thrown: it becomes the tool's result (`"error: ..."`), and the turn goes on.

## Example

```js
const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  try {
    // no org called "nobody" has a model provider
    const agent = await new AgentBuilder('nobody/no-model').build()
    await agent.close()
  } catch (/** @type {any} */ e) {
    if (e.code === 'AILOY_ERROR') console.error('ailoy refused:', e.message)
    else throw e
  }
}

main()
```

Guides: [Running an agent](/docs/guides/running/).
