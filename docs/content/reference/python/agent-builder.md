---
title: AgentBuilder
description: "Python reference for AgentBuilder: sets up an agent and builds it."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `ailoy-py`.*

Filled in place and emptied by ``build()``. Every setter returns this builder.

## `instruction`

```python
instruction(instruction: str) -> Self
```

## `tool`

```python
tool(desc: ToolDesc) -> Self
```

## `tools`

```python
tools(descs: list[ToolDesc]) -> Self
```

## `system_tools`

```python
system_tools() -> Self
```

## `shell_tool`

```python
shell_tool() -> Self
```

## `web_search_tool`

```python
web_search_tool(engines: list[WebSearchEngine] | None = None) -> Self
```

`engines` by name, `"Google"`, `"DuckDuckGo"`, …, or every engine when empty.

## `web_fetch_tool`

```python
web_fetch_tool() -> Self
```

## `history`

```python
history(history: list[Message]) -> Self
```

## `console`

```python
console(console: ConsoleClient) -> Self
```

Run the agent's console tools in the session of `console`, a `virtx.ConsoleClient`,
which the agent then shares.

## `browser`

```python
browser(
    *,
    user_browser: (
        Literal[
            "chrome",
            "chrome-beta",
            "chrome-dev",
            "chrome-canary",
            "chromium",
            "edge",
            "brave",
        ]
        | _Path
        | None
    ) = None,
    careful: bool = True,
    start_url: str | None = None,
    isolated: bool = False,
    headless: bool = False,
    executable: _Path | None = None,
    profile: _Path | None = None
) -> Self
```

A browser the agent drives with its ``browser_*`` tools over CDP from this process
rather than the console: the person's own, or with ``isolated`` one of the agent's own.

``isolated`` starts a browser on first use and closes it with the agent: in a window
unless ``headless``; ``executable`` if given, else a Chromium-based browser installed
here (Chrome, Chromium, Edge or Brave); on the user data directory ``profile`` if given,
so sign-ins last, else a fresh one.

The person's own needs remote debugging on (in Chrome 144 or later, the default, at
``chrome://inspect/#remote-debugging``), and asks the person before it lets the agent
in. ``user_browser`` names which, or gives any other's user data directory as a path;
with none, it is whichever turned remote debugging on last.

``careful`` tells the agent to use sites as a person would, so sites checking for bots
are less likely to turn it away.

``start_url`` is the page the agent's tab opens on.

## `memory`

```python
memory(memfile: str) -> Self
```

Remember into the memory store at `memfile`, a path in the console.

## `skill`

```python
skill(dir: str) -> Self
```

## `context_manager`

```python
context_manager(
    max_input_tokens: int | None = None,
    preserve_recent_turns: int | None = None,
    max_images: int | None = None,
    preserve_recent_images: int | None = None,
) -> Self
```

## `thinking_effort`

```python
thinking_effort(effort: ThinkingEffort) -> Self
```

How much the model thinks before it answers: `"low"`, `"medium"` or `"high"`.

## `output_schema`

```python
output_schema(schema: dict[str, Value]) -> Self
```

Constrain replies to the JSON schema `schema`.

## `build`

```python
async build() -> Agent
```

Make the agent, an awaitable, as `AgentBuilder::build` is a future: it builds the
model and reads the skills through the console.
