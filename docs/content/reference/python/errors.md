---
title: "Errors"
description: "How ailoy reports failures: what is raised, and where."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

ailoy raises three kinds of error:

- [`AiloyError`](#ailoyerror) for everything ailoy itself reports: a model whose org has no provider, a tool with no provider, an API error, an agent used during a turn or after `close()`.
- The `virtx` package's own classes for console failures, so a refusal inside a turn is the same `virtx.ConsoleRefused`, with the same `code`, as one from `ConsoleClient.exec`. A console whose channel is gone raises `virtx.ConsoleBroken`.
- `ValueError` for an argument ailoy can't take: a builder used after `build()`, an unknown `user_browser` name, a dict that isn't a message.

A turn's errors are raised from the `async for` that iterates it. A tool function's own exception isn't raised: it becomes the tool's result (`"error: ..."`), and the turn goes on.

## Example

```python
import asyncio

from ailoy import AgentBuilder, AiloyError


async def main() -> None:
    try:
        # no org called "nobody" has a model provider
        agent = await AgentBuilder("nobody/no-model").build()
    except AiloyError as e:
        print("ailoy refused:", e)
        return
    await agent.close()


asyncio.run(main())
```

Guides: [Running an agent](/docs/guides/running/).

## Members

| Member | Description |
| --- | --- |

