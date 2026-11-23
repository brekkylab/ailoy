---
title: "LangModel"
description: "A model run directly on a conversation, without an agent."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

A `LangModel` is a model on its own: no history, no turn loop, and no tools run. `create(desc)` builds it through the provider set for the desc's org, the way an agent builds its model, and `infer()` runs it once on the messages you pass.

When the model asks for a tool, the request comes back in the message's `tool_calls`. Running the tool, and sending its result back in the next call, is up to you. Use a `LangModel` for one-off calls, or as what a factory given to [`set_lang_model_provider`](../model-providers/) returns.

## Example

```python
import asyncio

from ailoy import LangModel
from ailoy.types import Message


async def main() -> None:
    model = await LangModel.create("anthropic/claude-sonnet-5-5")
    messages: list[Message] = [
        {"role": "system", "contents": [{"type": "text", "text": "Reply in French."}]},
        {"role": "user", "contents": [{"type": "text", "text": "Good morning!"}]},
    ]
    out = await model.infer(messages, options={"thinking_effort": "low"})
    for part in out["message"]["contents"]:
        if part["type"] == "text":
            print(part["text"])


asyncio.run(main())
```

Guides: [Model providers](/docs/guides/model-providers/).

## Members

| Member | Description |
| --- | --- |
| [`create()`](#create) | The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for `<org>`. |
| [`infer()`](#infer) | Run the model on `messages` to completion. |
| [`infer_stream()`](#infer_stream) | Run the model on `messages` and iterate its output as deltas while it is generated. |

## Methods

### `create()`

The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for
`<org>`. Awaitable.

```python
@staticmethod
async def create(desc: str) -> LangModel
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | `str` | required | The model's desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI's default model. See [Model providers](../model-providers/). |

**Returns** an awaitable of [`LangModel`](/docs/reference/python/lang-model/): The model.

**Raises** `AiloyError` when the org has no provider, or the backend can't build the model.

### `infer()`

Run the model on `messages` to completion. Awaitable; answers with the whole output.

```python
async def infer(
    messages: list[Message],
    *,
    tools: list[ToolDesc] | None = None,
    options: LangModelOptions | None = None
) -> MessageOutput
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | `list[`[`Message`](/docs/reference/python/messages/#message)`]` | required | The conversation to run the model on, oldest first. |
| `tools` | `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`] \| None` | `None` | The tools the model may call, by their descs. The model only asks for them: nothing runs them. |
| `options` | [`LangModelOptions`](/docs/reference/python/specs-and-options/#langmodeloptions)` \| None` | `None` | Thinking effort and output schema for this call. |

**Returns** an awaitable of [`MessageOutput`](/docs/reference/python/outputs/#messageoutput): The model's whole message, why it ended, and the tokens it took.

### `infer_stream()`

Run the model on `messages` and iterate its output as deltas while it is generated.

```python
def infer_stream(
    messages: list[Message],
    *,
    tools: list[ToolDesc] | None = None,
    options: LangModelOptions | None = None
) -> AgentRun[MessageDeltaOutput]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | `list[`[`Message`](/docs/reference/python/messages/#message)`]` | required | The conversation to run the model on, oldest first. |
| `tools` | `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`] \| None` | `None` | The tools the model may call, by their descs. The model only asks for them: nothing runs them. |
| `options` | [`LangModelOptions`](/docs/reference/python/specs-and-options/#langmodeloptions)` \| None` | `None` | Thinking effort and output schema for this call. |

**Returns** [`AgentRun`](/docs/reference/python/agent-run/)`[`[`MessageDeltaOutput`](/docs/reference/python/outputs/#messagedeltaoutput)`]`: The model's message as deltas.
