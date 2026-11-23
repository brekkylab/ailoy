---
title: LangModel
description: "Python reference for LangModel: a model run without an agent."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `ailoy-py`.*

A model built from its desc, as an ``Agent`` builds one, run directly.

## `create`

```python
@staticmethod
async create(desc: str) -> LangModel
```

The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for
`<org>`. Awaitable.

## `infer`

```python
async infer(
    messages: list[Message],
    *,
    tools: list[ToolDesc] | None = None,
    options: LangModelOptions | None = None
) -> MessageOutput
```

Run the model on `messages` to completion. Awaitable; answers with the whole output.

## `infer_stream`

```python
infer_stream(
    messages: list[Message],
    *,
    tools: list[ToolDesc] | None = None,
    options: LangModelOptions | None = None
) -> AgentRun[MessageDeltaOutput]
```

Run the model on `messages` and iterate its output as deltas while it is generated.
