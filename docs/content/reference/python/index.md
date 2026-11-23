---
title: "Python installation"
description: "Install ailoy-py, import it, and see how its classes fit together."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

This page installs `ailoy-py` and shows how its classes fit together. New here? Start with the [introduction](/docs/reference/).

## Install

```sh
pip install ailoy-py
```

It needs Python 3.10 or later. Every class and function is imported from `ailoy`, and the dict shapes from `ailoy.types`:

```python
from ailoy import AgentBuilder
```

## How the pieces fit

[`AgentBuilder`](agent-builder/) configures an agent, and `await builder.build()` makes an [`Agent`](agent/). `agent.run(query)` returns an [`AgentRun`](agent-run/), the turn, which you iterate with `async for` to get each [`MessageOutput`](outputs/#messageoutput). `run_stream` gives [`MessageDeltaOutput`](outputs/#messagedeltaoutput)s as the model writes.

Models and tools are set up process-wide, before an agent is built: [Model providers](model-providers/) decide how each model runs, and [Tools and MCP](tools/) register what runs behind each tool name. The [data shapes](messages/) are plain dicts in their JSON form.

## Pages

### Get started

| Page | What it covers |
| --- | --- |
| [Errors](errors/) | How ailoy reports failures: what is raised, and where. |

### Agents

| Page | What it covers |
| --- | --- |
| [AgentBuilder](agent-builder/) | Configures an agent, its model, instruction, tools, console, browser, memory and skills, and builds it. |
| [Agent](agent/) | A built agent: runs turns on a query and keeps the conversation. |
| [AgentRun](agent-run/) | One turn of an agent, iterated as it runs. |

### Models

| Page | What it covers |
| --- | --- |
| [LangModel](lang-model/) | A model run directly on a conversation, without an agent. |
| [Model providers](model-providers/) | How a model desc is built: the backend set for each org, or a model of your own. |

### Tools

| Page | What it covers |
| --- | --- |
| [Tools and MCP](tools/) | Registering your own functions and MCP servers as tools an agent can call. |

### Data shapes

| Page | What it covers |
| --- | --- |
| [Messages](messages/) | The shape of a message and its parts: text, tool calls, values and images. |
| [Outputs](outputs/) | What a turn or a model call gives back: whole messages, streamed deltas, finish reasons and token usage. |
| [Specs and options](specs-and-options/) | An agent as data, and the options a model, a browser and the context manager take. |
