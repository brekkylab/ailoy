---
title: "AgentBuilder"
description: "Configures an agent, its model, instruction, tools, console, browser, memory and skills, and builds it."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

`AgentBuilder` collects everything an agent is made of, then `build()` turns it into an [`Agent`](../agent/). Start one with the model's desc, call the methods for what the agent needs, and await `build()`.

Each method changes the builder in place and returns it, so calls chain. Nothing is checked or started until `build()`. That's when the model is built through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` is read through the console. A missing provider raises [`AiloyError`](../errors/) from `build()`, and a `SKILL.md` the console can't read raises the console's own failure, such as `virtx.ConsoleRefused`.

A builder makes one agent. Once `build()` has run, every method raises `ValueError` ("this AgentBuilder has already been built").

## Example

```python
import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    agent = await (
        AgentBuilder("anthropic/claude-sonnet-5-5")
        .instruction("You are a research assistant. Name the pages you read.")
        .web_search_tool(["DuckDuckGo", "Bing"])
        .web_fetch_tool()
        .thinking_effort("medium")
        .context_manager(max_input_tokens=100_000)
        .build()
    )
    async with agent:
        async for out in agent.run("What changed in the latest Python release?"):
            for part in out["message"]["contents"]:
                if part["type"] == "text":
                    print(part["text"])


asyncio.run(main())
```

Guides: [Defining an agent](/docs/guides/agents/), [Tools](/docs/guides/tools/), [Giving an agent a sandbox](/docs/guides/sandbox/), [Skills](/docs/guides/skills/).

## Members

| Member | Description |
| --- | --- |
| [`AgentBuilder()`](#agentbuilder) | A builder for the model `model` names, built through the provider set for its org when `build()` runs. |
| [`instruction()`](#instruction) | Set the system instruction: the system message the agent's history starts with, which shapes how the model works. |
| [`tool()`](#tool) | Give the agent one tool, by its desc. |
| [`tools()`](#tools) | Give the agent several tools at once, such as the descs `register_mcp_stdio` hands back. |
| [`system_tools()`](#system_tools) | Give the agent the local-execution tools that suit its model's family: `shell`; for `openai`, `apply_patch`; for every other org, `read` (`read_file` for `google`), `write` and `edit`; and `imgread`, except for `deepseek-ai`, `moonshotai`, `zai-org` and `Qwen`, whose backends take tool results as text alone. |
| [`shell_tool()`](#shell_tool) | Give the agent only the `shell` tool, without the rest of `system_tools()`. |
| [`web_search_tool()`](#web_search_tool) | Give the agent the `web_search` tool, which searches the web through the engines named, or through every engine when none are. |
| [`web_fetch_tool()`](#web_fetch_tool) | Give the agent the `web_fetch` tool, which isn't part of `system_tools()`. |
| [`history()`](#history) | Start the agent with this conversation, for example to resume an earlier session. |
| [`console()`](#console) | Run the agent's console tools in the session of `console`, a `virtx.ConsoleClient`, which the agent then shares. |
| [`browser()`](#browser) | A browser the agent drives with its `browser_*` tools over CDP from this process rather than the console: the person's own, or with `isolated` one of the agent's own. |
| [`memory()`](#memory) | Remember into the memory store at `memfile`, a path in the console. |
| [`skill()`](#skill) | Give the agent the skill in `dir`, a directory in the console holding a `SKILL.md`. |
| [`context_manager()`](#context_manager) | Set how the agent keeps its history within the model's context: truncating old tool results once the input grows past a limit, and capping the images tool results hold. |
| [`thinking_effort()`](#thinking_effort) | How much the model thinks before it answers: `"low"`, `"medium"` or `"high"`. |
| [`output_schema()`](#output_schema) | Constrain replies to the JSON schema `schema`. |
| [`build()`](#build) | Build the agent: the model through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` read through the console. |

## Constructor

### `AgentBuilder()`

A builder for the model `model` names, built through the provider set for its org when `build()` runs.

```python
AgentBuilder(model: str)
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | required | The model's desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI's default model. See [Model providers](../model-providers/). |

## Methods

### `instruction()`

Set the system instruction: the system message the agent's history starts with, which shapes how the model works. A leading system message in `history()` replaces it.

```python
def instruction(instruction: str) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `instruction` | `str` | required | The system prompt. |

**Returns** this builder, for chaining.

### `tool()`

Give the agent one tool, by its desc. The model sees the desc; `build()` builds the tool through the provider set for `desc.name`.

```python
def tool(desc: ToolDesc) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | [`ToolDesc`](/docs/reference/python/tools/#tooldesc) | required | The tool's name, description and parameter schema, such as what `register_tool` hands back. |

**Returns** this builder, for chaining.

### `tools()`

Give the agent several tools at once, such as the descs `register_mcp_stdio` hands back.

```python
def tools(descs: list[ToolDesc]) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `descs` | `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`]` | required | The tools' descs, each built through the provider set for its name. |

**Returns** this builder, for chaining.

### `system_tools()`

Give the agent the local-execution tools that suit its model's family: `shell`; for `openai`, `apply_patch`; for every other org, `read` (`read_file` for `google`), `write` and `edit`; and `imgread`, except for `deepseek-ai`, `moonshotai`, `zai-org` and `Qwen`, whose backends take tool results as text alone. These tools run in the console, so the agent needs `console()` too.

```python
def system_tools() -> Self
```

**Returns** this builder, for chaining.

### `shell_tool()`

Give the agent only the `shell` tool, without the rest of `system_tools()`. It runs in the console, so the agent needs `console()` too.

```python
def shell_tool() -> Self
```

**Returns** this builder, for chaining.

### `web_search_tool()`

Give the agent the `web_search` tool, which searches the web through the engines named, or through every engine when none are.

```python
def web_search_tool(engines: list[WebSearchEngine] | None = None) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `engines` | `list[`[`WebSearchEngine`](/docs/reference/python/tools/#websearchengine)`] \| None` | `None` | The engines to search with. All of them when empty or left out. |

**Returns** this builder, for chaining.

### `web_fetch_tool()`

Give the agent the `web_fetch` tool, which isn't part of `system_tools()`. It fetches one URL per call, pages long bodies through `offset`, and allows one request per second per host.

```python
def web_fetch_tool() -> Self
```

**Returns** this builder, for chaining.

### `history()`

Start the agent with this conversation, for example to resume an earlier session. A leading system message here replaces the one `instruction()` would make; otherwise the instruction is put at the front of it.

```python
def history(history: list[Message]) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `history` | `list[`[`Message`](/docs/reference/python/messages/#message)`]` | required | The earlier messages, oldest first, such as a previous agent's `history`. |

**Returns** this builder, for chaining.

### `console()`

Run the agent's console tools in the session of `console`, a `virtx.ConsoleClient`,
which the agent then shares.

The console needn't be started: the first batch of tool calls with a console tool starts it. Nothing makes a console for you. Without one, tools that don't need it still run, and console tools fail with an error.

```python
def console(console: ConsoleClient) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `console` | `ConsoleClient` | required | A `virtx.ConsoleClient`. |

**Returns** this builder, for chaining.

### `browser()`

A browser the agent drives with its `browser_*` tools over CDP from this process
rather than the console: the person's own, or with `isolated` one of the agent's own.

`isolated` starts a browser on first use and closes it with the agent: in a window
unless `headless`; `executable` if given, else a Chromium-based browser installed
here (Chrome, Chromium, Edge or Brave); on the user data directory `profile` if given,
so sign-ins last, else a fresh one.

The person's own needs remote debugging on (in Chrome 144 or later, the default, at
`chrome://inspect/#remote-debugging`), and asks the person before it lets the agent
in. `user_browser` names which, or gives any other's user data directory as a path;
with none, it is whichever turned remote debugging on last.

`careful` tells the agent to use sites as a person would, so sites checking for bots
are less likely to turn it away.

`start_url` is the page the agent's tab opens on.

Nothing starts until the agent first uses the browser, or `Agent.start_browser()`.

```python
def browser(
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
        | str
        | os.PathLike[str]
        | None
    ) = None,
    careful: bool = True,
    start_url: str | None = None,
    isolated: bool = False,
    headless: bool = False,
    executable: str | os.PathLike[str] | None = None,
    profile: str | os.PathLike[str] | None = None
) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `user_browser` | `Literal['chrome', 'chrome-beta', 'chrome-dev', 'chrome-canary', 'chromium', 'edge', 'brave'] \| str \| os.PathLike[str] \| None` | `None` | Without `isolated`: which of the person's browsers, by name, or any other Chromium-based browser by its user data directory (a path). With none, whichever turned remote debugging on last. |
| `careful` | `bool` | `True` | Tell the agent to use sites as a person would, so sites checking for bots are less likely to turn it away. |
| `start_url` | `str \| None` | `None` | The page the agent's tab opens on. |
| `isolated` | `bool` | `False` | Start a browser of the agent's own, apart from the person's: started on first use and closed with the agent. Off by default, which drives the person's own browser. |
| `headless` | `bool` | `False` | With `isolated`: run it with no window, as on a server. |
| `executable` | `str \| os.PathLike[str] \| None` | `None` | With `isolated`: the Chromium-based browser to start, by its executable. By default one installed here (Chrome, Chromium, Edge or Brave). |
| `profile` | `str \| os.PathLike[str] \| None` | `None` | With `isolated`: the user data directory it keeps, so sign-ins last. A fresh one by default. |

**Returns** this builder, for chaining.

### `memory()`

Remember into the memory store at `memfile`, a path in the console.

Adds the `mem_search` and `mem_insert` tools. The store must already exist (`mem init`): a missing file is reported by the first memory tool call, not by `build()`. Memory tools run in the console, so the agent needs `console()` too.

```python
def memory(memfile: str) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `memfile` | `str` | required | The memory store's path in the console. |

**Returns** this builder, for chaining.

### `skill()`

Give the agent the skill in `dir`, a directory in the console holding a `SKILL.md`. At `build()`, its frontmatter `name` and `description` are read through the console and listed in the system message with the skill's location. The agent reads the rest itself when it needs it, so it needs a console and a file-reading tool. Skills are only added to the system message `instruction()` makes, never to one already in `history()`. Call it once per skill.

```python
def skill(dir: str) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dir` | `str` | required | The skill's directory, in the console. |

**Returns** this builder, for chaining.

### `context_manager()`

Set how the agent keeps its history within the model's context: truncating old tool results once the input grows past a limit, and capping the images tool results hold. Settings left out keep their defaults. Without this call the defaults apply anyway: no truncation, and the image cap.

```python
def context_manager(
    max_input_tokens: int | None = None,
    preserve_recent_turns: int | None = None,
    max_images: int | None = None,
    preserve_recent_images: int | None = None,
) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `max_input_tokens` | `int \| None` | `None` | Truncate the history as a turn begins once the previous model call's input went past this many tokens. `0` turns truncation off. Truncation keeps the system message and the recent turns, and replaces each earlier tool result's contents with `"[context truncated]"`, keeping its `id`. Whole turns are never dropped. |
| `preserve_recent_turns` | `int \| None` | `None` | How many recent user turns truncation leaves alone. The turn just begun is one of them, so `0` keeps it too. |
| `max_images` | `int \| None` | `None` | How many images tool results may hold in the history at the start of a turn. Past it, the oldest become `"[image from an earlier turn removed]"`. Images a person attached are left alone. `0` turns the cap off. |
| `preserve_recent_images` | `int \| None` | `None` | How many tool-result images are left once `max_images` is passed. They're removed several at a time, so the prompt cache breaks once in several turns rather than every turn. |

**Returns** this builder, for chaining.

### `thinking_effort()`

How much the model thinks before it answers: `"low"`, `"medium"` or `"high"`.

Left unset, the model or backend default applies.

```python
def thinking_effort(effort: ThinkingEffort) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `effort` | [`ThinkingEffort`](/docs/reference/python/specs-and-options/#thinkingeffort) | required | `"low"`, `"medium"` or `"high"`. |

**Returns** this builder, for chaining.

### `output_schema()`

Constrain replies to the JSON schema `schema`.

```python
def output_schema(schema: dict[str, Value]) -> Self
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `schema` | `dict[str, `[`Value`](/docs/reference/python/messages/#value)`]` | required | A JSON schema object. |

**Returns** this builder, for chaining.

### `build()`

Build the agent: the model through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` read through the console. Awaitable, since building the model and reading skills take time.

```python
async def build() -> Agent
```

**Returns** an awaitable of [`Agent`](/docs/reference/python/agent/): The agent.

**Raises** `AiloyError` when the model's org or a tool has no provider, a skill is given without a console or its frontmatter can't be read, or the browser settings don't fit together. The console's own failure, such as `virtx.ConsoleRefused`, when a skill's `SKILL.md` can't be read. `ValueError` when the builder was already built.
