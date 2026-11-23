---
title: Functions
description: "Python reference for the module functions: model and tool providers, and MCP."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `ailoy-py`.*

## `get_lang_model_provider`

```python
get_lang_model_provider(desc: str) -> Backend | Literal['custom'] | None
```

How the model `desc` names (`"<org>/<name>"`, or `"<org>"`) is built: a backend's name,
`"custom"` for a factory, or `None` when its org has no provider.

## `register_mcp_stdio`

```python
async register_mcp_stdio(
    prefix: str, command: str, args: list[str] = ...
) -> list[ToolDesc]
```

Start the MCP server `command args…` on this host and register each of its tools as
`{prefix}__{name}`. Awaitable; answers with their descriptions.

The server runs outside any console, with this process's access, register only servers
you trust.

## `register_mcp_streamable_http`

```python
async register_mcp_streamable_http(prefix: str, url: str) -> list[ToolDesc]
```

Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
`{prefix}__{name}`. Awaitable; answers with their descriptions.

## `register_tool`

```python
register_tool(
    desc: ToolDesc, func: Callable[..., Any | Awaitable[Any]]
) -> ToolDesc
```

Build the tool `desc` describes by calling `func`, and hand `desc` back.

`desc` is `{"name": ..., "description": ..., "parameters": <JSON schema>}`; the name is
what the model calls and what the provider is set under.

## `remove_lang_model_provider`

```python
remove_lang_model_provider(org: str) -> bool
```

Stop building the models of `org`. Answers whether it had a provider.

## `remove_tool_provider`

```python
remove_tool_provider(name: str) -> bool
```

Stop building the tool `name`. Answers whether it had a provider.

## `set_lang_model_provider`

```python
set_lang_model_provider(org: str, provider: Backend | LangModelFactory) -> None
```

Build the models of `org` (`"anthropic"`, `"openai"`, `"google"`, `"deepseek-ai"`, …)
through `provider`: a backend, `"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or `"vllm"`, or a factory, as the module says. Replaces the provider the org had.

## `set_tool_provider`

```python
set_tool_provider(name: str, func: Callable[..., Any | Awaitable[Any]]) -> None
```

Build the tool `name` by calling `func`, replacing the provider it had. See
[`crate::tool`](https://docs.rs/ailoy/0.3.2/ailoy/tool/index.html) for how `func` is called.

## `unregister_mcp`

```python
unregister_mcp(prefix: str) -> int
```

Drop the tools registered under `prefix` and end that server's session. Answers with how
many were removed.
