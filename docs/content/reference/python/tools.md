---
title: "Tools and MCP"
description: "Registering your own functions and MCP servers as tools an agent can call."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

A tool has two halves. Its [`ToolDesc`](#tooldesc), the name, description and parameter schema, is what the model sees, and it goes on the agent with `AgentBuilder.tool()` or `tools()`. Its provider, what runs when the model calls it, is set process-wide under the tool's name. When an agent is built, each desc is built through the provider set for its name, and a name with none fails `build()`.

The registry starts with every built-in tool (`shell`, `read`, `write`, `edit`, `apply_patch`, `web_search`, `web_fetch` and the rest), which the builder's `system_tools()`, `shell_tool()`, `web_search_tool()` and `web_fetch_tool()` add. `register_tool()` adds a function of your own, and `register_mcp_stdio()` and `register_mcp_streamable_http()` add an MCP server's tools. Each of these hands back the descs to give the agent.

A Python tool function is called with the model's arguments as keyword arguments when they're a dict, or else as one positional argument. It can be async. It returns what a message can hold: `None`, a bool, a number, a string, or a list or dict of those. A raise doesn't end the turn: it becomes the tool's result, `"error: ValueError: ..."`, so the model can try differently. A tool function isn't handed the agent's console; one that runs commands holds its own `ConsoleClient`.

## Example

```python
import asyncio

from ailoy import AgentBuilder, register_mcp_stdio, register_tool


def get_weather(city: str) -> str:
    return f"Sunny and 21°C in {city}"


# Your own function: the desc is what the model sees, the function what runs
weather = register_tool(
    {
        "name": "get_weather",
        "description": "The current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    get_weather,
)


async def main() -> None:
    # An MCP server's tools, named fs__<tool>
    fs_tools = await register_mcp_stdio(
        "fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    agent = await AgentBuilder("anthropic/claude-sonnet-5-5").tool(weather).tools(fs_tools).build()
    async with agent:
        async for out in agent.run("What's the weather in Seoul?"):
            print(out["message"]["role"], out["message"]["contents"])


asyncio.run(main())
```

Guides: [Tools](/docs/guides/tools/), [Using MCP](/docs/guides/using-mcp/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`register_tool()`](#register_tool) | function | Build the tool `desc` describes by calling `func`, and hand `desc` back. |
| [`set_tool_provider()`](#set_tool_provider) | function | Build the tool `name` by calling `func`, replacing the provider it had. |
| [`remove_tool_provider()`](#remove_tool_provider) | function | Stop building the tool `name`. |
| [`register_mcp_stdio()`](#register_mcp_stdio) | function | Start the MCP server `command args…` on this host and register each of its tools as `{prefix}__{name}`. |
| [`register_mcp_streamable_http()`](#register_mcp_streamable_http) | function | Connect to the streamable-HTTP MCP server at `url` and register each of its tools as `{prefix}__{name}`. |
| [`unregister_mcp()`](#unregister_mcp) | function | Drop the tools registered under `prefix` and end that server's session. |
| [`ToolDesc`](#tooldesc) | type | A tool as the model sees it. |
| [`WebSearchEngine`](#websearchengine) | type | An engine the `web_search` tool can use. |

## `register_tool()`

Build the tool `desc` describes by calling `func`, and hand `desc` back.

`desc` is `{"name": ..., "description": ..., "parameters": <JSON schema>}`; the name is
what the model calls and what the provider is set under.

```python
def register_tool(
    desc: ToolDesc, func: Callable[..., Any | Awaitable[Any]]
) -> ToolDesc
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | [`ToolDesc`](/docs/reference/python/tools/#tooldesc) | required | The tool's name, description and parameter schema. `desc.name` is what the model calls. |
| `func` | `Callable[..., Any \| Awaitable[Any]]` | required | What runs when the model calls the tool: a function, sync or async, called with the model's arguments as keyword arguments. |

**Returns** [`ToolDesc`](/docs/reference/python/tools/#tooldesc): The same `desc`, to give the agent.

## `set_tool_provider()`

Build the tool `name` by calling `func`, replacing the provider it had. See
[`crate::tool`](https://docs.rs/ailoy/0.3.2/ailoy/tool/index.html) for how `func` is called.

```python
def set_tool_provider(
    name: str, func: Callable[..., Any | Awaitable[Any]]
) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | The tool's name. |
| `func` | `Callable[..., Any \| Awaitable[Any]]` | required | What runs when the model calls the tool: a function, sync or async, called with the model's arguments as keyword arguments. |

## `remove_tool_provider()`

Stop building the tool `name`. Answers whether it had a provider.

An agent built afterwards with the tool fails to build.

```python
def remove_tool_provider(name: str) -> bool
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | The tool's name. |

**Returns** `bool`: Whether the tool had a provider.

## `register_mcp_stdio()`

Start the MCP server `command args…` on this host and register each of its tools as
`{prefix}__{name}`. Awaitable; answers with their descriptions.

The server runs outside any console, with this process's access, register only servers
you trust.

```python
async def register_mcp_stdio(
    prefix: str, command: str, args: list[str] = ...
) -> list[ToolDesc]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `str` | required | Put before each tool's name, as `{prefix}__{name}`. It's also how `unregister_mcp` names the server. |
| `command` | `str` | required | The server's executable. |
| `args` | `list[str]` | none | The command's arguments. |

**Returns** an awaitable of `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`]`: The descs of the server's tools, to give the agent with `tools()`.

## `register_mcp_streamable_http()`

Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
`{prefix}__{name}`. Awaitable; answers with their descriptions.

```python
async def register_mcp_streamable_http(
    prefix: str, url: str
) -> list[ToolDesc]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `str` | required | Put before each tool's name, as `{prefix}__{name}`. It's also how `unregister_mcp` names the server. |
| `url` | `str` | required | The server's streamable-HTTP endpoint. |

**Returns** an awaitable of `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`]`: The descs of the server's tools, to give the agent with `tools()`.

## `unregister_mcp()`

Drop the tools registered under `prefix` and end that server's session. Answers with how
many were removed.

```python
def unregister_mcp(prefix: str) -> int
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `str` | required | The prefix the server was registered under. |

**Returns** `int`: How many tools were removed.

## `ToolDesc`

A tool as the model sees it.

```python
class ToolDesc(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Value
    returns: NotRequired[Value]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | `str` | yes | The tool's name: what the model calls, and what its provider is set under. |
| `description` | `str` | no | What the tool does, in words, for the model. |
| `parameters` | [`Value`](/docs/reference/python/messages/#value) | yes | A JSON schema of the arguments, usually an object schema: `{"type": "object", "properties": ...}`. |
| `returns` | [`Value`](/docs/reference/python/messages/#value) | no | A JSON schema of the result. Left out, the tool returns free-form text or JSON. |

## `WebSearchEngine`

An engine the `web_search` tool can use.

```python
WebSearchEngine = Literal[
    "Bing",
    "Brave",
    "DuckDuckGo",
    "Google",
    "Mojeek",
    "Naver",
    "Startpage",
    "Yahoo",
    "Yandex",
]
```
