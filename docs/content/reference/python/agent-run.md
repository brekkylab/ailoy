---
title: "AgentRun"
description: "One turn of an agent, iterated as it runs."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

`Agent.run()` and `Agent.run_stream()` return an `AgentRun`, the turn before it has started. `LangModel.infer_stream()` returns one too. Iterate it with `async for`.

Nothing runs until the first item is asked for, and the turn only goes as far as it has been iterated. Breaking out of the loop, or `aclose()`, ends the turn where it stands.

What it yields depends on where it came from. `run()` yields a [`MessageOutput`](../outputs/#messageoutput) per complete message: the model's, then each tool's result. `run_stream()` and `infer_stream()` yield [`MessageDeltaOutput`](../outputs/#messagedeltaoutput)s: pieces of the model's message as it's generated, and each tool's result whole. Errors from the model, a tool's provider or the console are raised from the loop.

## Example

```python
import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").web_fetch_tool().build() as agent:
        turn = agent.run("Read https://example.com and summarize it.")  # nothing has run yet
        async for out in turn:
            print(out["message"]["role"], out["finish_reason"]["type"])
            if out["message"]["role"] == "tool":
                break  # ends the turn here, after the first tool result


asyncio.run(main())
```

Guides: [Running an agent](/docs/guides/running/).

## Members

| Member | Description |
| --- | --- |
| [`aclose()`](#aclose) | End the turn where it stands, as breaking out of `async for` does. |

## Methods

### `aclose()`

End the turn where it stands, as breaking out of `async for` does.

```python
async def aclose() -> None
```
