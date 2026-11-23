---
title: "Tools and MCP"
description: "Registering your own functions and MCP servers as tools an agent can call."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

A tool has two halves. Its [`ToolDesc`](#tooldesc), the name, description and parameter schema, is what the model sees, and it goes on the agent with `AgentBuilder.tool()` or `tools()`. Its provider, what runs when the model calls it, is set process-wide under the tool's name. When an agent is built, each desc is built through the provider set for its name, and a name with none rejects `build()`.

The registry starts with every built-in tool (`shell`, `read`, `write`, `edit`, `apply_patch`, `web_search`, `web_fetch` and the rest), which the builder's `systemTools()`, `shellTool()`, `webSearchTool()` and `webFetchTool()` add. `registerTool()` adds a function of your own, and `registerMcpStdio()` and `registerMcpStreamableHttp()` add an MCP server's tools. Each of these hands back the descs to give the agent.

A JavaScript tool function is called with the model's arguments, the object the desc's `parameters` schema describes. It can return a value or a promise of one. A throw or a rejection doesn't end the turn: it becomes the tool's result, `"error: ..."`, so the model can try differently.

## Example

```js
const { AgentBuilder, registerMcpStdio, registerTool } = require('@brekkylab/ailoy')

// Your own function: the desc is what the model sees, the function what runs
const weather = registerTool(
  {
    name: 'get_weather',
    description: 'The current weather in a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
  },
  ({ city }) => `Sunny and 21°C in ${city}`,
)

async function main() {
  // An MCP server's tools, named fs__<tool>
  const fsTools = await registerMcpStdio('fs', 'npx', ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])

  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').tool(weather).tools(fsTools).build()
  for await (const out of agent.run("What's the weather in Seoul?")) {
    console.log(out.message.role, out.message.contents)
  }
  await agent.close()
}

main()
```

Guides: [Tools](/docs/guides/tools/), [Using MCP](/docs/guides/using-mcp/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`registerTool()`](#registertool) | function | Build the tool `desc` describes by calling `func`, and hand `desc` back. |
| [`setToolProvider()`](#settoolprovider) | function | Build the tool `name` by calling `func`, replacing the provider it had. |
| [`removeToolProvider()`](#removetoolprovider) | function | Stop building the tool `name`. |
| [`registerMcpStdio()`](#registermcpstdio) | function | Start the MCP server `command args…` on this host and register each of its tools as `{prefix}__{name}`, settling with their descriptions. |
| [`registerMcpStreamableHttp()`](#registermcpstreamablehttp) | function | Connect to the streamable-HTTP MCP server at `url` and register each of its tools as `{prefix}__{name}`, settling with their descriptions. |
| [`unregisterMcp()`](#unregistermcp) | function | Drop the tools registered under `prefix` and end that server's session. |
| [`ToolDesc`](#tooldesc) | type | A tool as the model sees it. |
| [`WebSearchEngine`](#websearchengine) | type | An engine the `web_search` tool can use. |

## `registerTool()`

Build the tool `desc` describes by calling `func`, and hand `desc` back.

`desc.name` is what the model calls and what the provider is set under.

```ts
function registerTool(desc: ToolDesc, func: (args: any) => unknown): ToolDesc
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc) | required | The tool's name, description and parameter schema. `desc.name` is what the model calls. |
| `func` | `(args: any) => unknown` | required | What runs when the model calls the tool: a function called with the model's arguments object, returning a value or a promise. |

**Returns** [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc): The same `desc`, to give the agent.

## `setToolProvider()`

Build the tool `name` by calling `func`, replacing the provider it had. See [`crate::tool`](https://docs.rs/ailoy/0.3.2/ailoy/tool/index.html)
for how `func` is called.

```ts
function setToolProvider(name: string, func: (args: any) => unknown): void
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `string` | required | The tool's name. |
| `func` | `(args: any) => unknown` | required | What runs when the model calls the tool: a function called with the model's arguments object, returning a value or a promise. |

## `removeToolProvider()`

Stop building the tool `name`. Answers whether it had a provider.

An agent built afterwards with the tool fails to build.

```ts
function removeToolProvider(name: string): boolean
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `string` | required | The tool's name. |

**Returns** `boolean`: Whether the tool had a provider.

## `registerMcpStdio()`

Start the MCP server `command args…` on this host and register each of its tools as
`{prefix}__{name}`, settling with their descriptions.

The server runs outside any console, with this process's access, register only servers
you trust.

```ts
function registerMcpStdio(
  prefix: string,
  command: string,
  args?: string[] | null,
): Promise<ToolDesc[]>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `string` | required | Put before each tool's name, as `{prefix}__{name}`. It's also how `unregister_mcp` names the server. |
| `command` | `string` | required | The server's executable. |
| `args` | `string[] \| null` | none | The command's arguments. |

**Returns** `Promise<`[`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]>`: The descs of the server's tools, to give the agent with `tools()`.

## `registerMcpStreamableHttp()`

Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
`{prefix}__{name}`, settling with their descriptions.

```ts
function registerMcpStreamableHttp(
  prefix: string,
  url: string,
): Promise<ToolDesc[]>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `string` | required | Put before each tool's name, as `{prefix}__{name}`. It's also how `unregister_mcp` names the server. |
| `url` | `string` | required | The server's streamable-HTTP endpoint. |

**Returns** `Promise<`[`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]>`: The descs of the server's tools, to give the agent with `tools()`.

## `unregisterMcp()`

Drop the tools registered under `prefix` and end that server's session. Answers with how
many were removed.

```ts
function unregisterMcp(prefix: string): number
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `string` | required | The prefix the server was registered under. |

**Returns** `number`: How many tools were removed.

## `ToolDesc`

A tool as the model sees it.

```ts
interface ToolDesc {
  name: string;
  description?: string;
  parameters: any;
  returns?: any;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | `string` | yes | The tool's name: what the model calls, and what its provider is set under. |
| `description` | `string` | no | What the tool does, in words, for the model. |
| `parameters` | `any` | yes | A JSON schema of the arguments, usually an object schema: `{"type": "object", "properties": ...}`. |
| `returns` | `any` | no | A JSON schema of the result. Left out, the tool returns free-form text or JSON. |

## `WebSearchEngine`

An engine the `web_search` tool can use.

```ts
type WebSearchEngine = "Bing" | "Brave" | "DuckDuckGo" | "Google" | "Mojeek" | "Naver" | "Startpage" | "Yahoo" | "Yandex";
```
