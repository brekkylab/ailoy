---
title: "Specs and options"
description: "An agent as data, and the options a model, a browser and the context manager take."
next: false
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

[`AgentSpec`](#agentspec) is what makes an agent distinct (its model, instruction, tools, model options, search engines and skills) as one dict, for `Agent.from_spec`. [`LangModelOptions`](#langmodeloptions) are the settings every model call takes, as `LangModel.infer()` and a custom model receive them.

In Python, the browser and context manager settings are keyword arguments of `AgentBuilder.browser()` and `AgentBuilder.context_manager()`, documented there.

## Example

```python
import asyncio
import json

from ailoy import Agent, AgentBuilder
from ailoy.types import AgentSpec

# An agent as data: store it as JSON, load it, build it
spec: AgentSpec = {
    "model": "anthropic/claude-sonnet-5-5",
    "instruction": "Be brief.",
    "model_options": {"thinking_effort": "low"},
}
saved = json.dumps(spec)


async def main() -> None:
    agent = await Agent.from_spec(json.loads(saved))
    await agent.close()

    # In Python, the browser and context settings are the builder's keyword arguments
    browsing = await (
        AgentBuilder("anthropic/claude-sonnet-5-5")
        .browser(isolated=True, headless=True)
        .context_manager(max_input_tokens=100_000, preserve_recent_turns=2)
        .build()
    )
    await browsing.close()


asyncio.run(main())
```

Guides: [Defining an agent](/docs/guides/agents/), [Context manager](/docs/guides/context-manager/), [Using a browser](/docs/guides/using-browser/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`AgentSpec`](#agentspec) | type | What makes an agent distinct, as data: its model, instruction and tools. |
| [`LangModelOptions`](#langmodeloptions) | type | The settings every model call takes, whichever backend runs it. |
| [`ThinkingEffort`](#thinkingeffort) | type | How much the model thinks before it answers. |

## `AgentSpec`

What makes an agent distinct, as data: its model, instruction and tools. The console, browser, memory and history aren't part of it.

```python
class AgentSpec(TypedDict):
    model: str
    instruction: NotRequired[str]
    tools: NotRequired[list[ToolDesc]]
    model_options: NotRequired[LangModelOptions]
    web_search_engines: NotRequired[list[str]]
    skills: NotRequired[list[str]]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | `str` | yes | The model's desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI's default model. See [Model providers](../model-providers/). |
| `instruction` | `str` | no | The system instruction. |
| `tools` | `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`]` | no | The tools' descs. Each name needs a provider set for it. |
| `model_options` | [`LangModelOptions`](/docs/reference/python/specs-and-options/#langmodeloptions) | no | Thinking effort and output schema. |
| `web_search_engines` | `list[str]` | no | The engines `web_search` uses, when it's among `tools`. All of them when left out. |
| `skills` | `list[str]` | no | Skill directories in the console, each holding a `SKILL.md`. |

## `LangModelOptions`

The settings every model call takes, whichever backend runs it.

```python
class LangModelOptions(TypedDict):
    thinking_effort: NotRequired[ThinkingEffort]
    output_schema: NotRequired[Value]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `thinking_effort` | [`ThinkingEffort`](/docs/reference/python/specs-and-options/#thinkingeffort) | no | How much the model thinks. Left out, the model or backend default. |
| `output_schema` | [`Value`](/docs/reference/python/messages/#value) | no | A JSON schema the output must match. |

## `ThinkingEffort`

How much the model thinks before it answers.

```python
ThinkingEffort = Literal["low", "medium", "high"]
```
