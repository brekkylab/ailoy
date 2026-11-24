---
title: "AgentBuilder"
description: "Configures an agent, its model, instruction, tools, console, browser, memory and skills, and builds it."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

`AgentBuilder` collects everything an agent is made of, then `build()` turns it into an [`Agent`](../agent/). Start one with `new AgentBuilder(model)`, call the methods for what the agent needs, and await `build()`.

Each method changes the builder in place and returns it, so calls chain. Nothing is checked or started until `build()`. That's when the model is built through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` is read through the console. A missing provider rejects `build()` with an [`AILOY_ERROR`](../errors/), and a `SKILL.md` the console can't read rejects it with the console's code, such as `NOT_FOUND`.

A builder makes one agent. Once `build()` has run, every method throws an error with code `INVALID_ARG` ("this AgentBuilder has already been built").

## Example

```js
const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5')
    .instruction('You are a research assistant. Name the pages you read.')
    .webSearchTool(['DuckDuckGo', 'Bing'])
    .webFetchTool()
    .thinkingEffort('medium')
    .contextManager({ maxInputTokens: 100_000 })
    .build()

  for await (const out of agent.run('What changed in the latest Python release?')) {
    for (const part of out.message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
  await agent.close()
}

main()
```

Guides: [Defining an agent](/docs/guides/agents/), [Tools](/docs/guides/tools/), [Giving an agent a sandbox](/docs/guides/sandbox/), [Skills](/docs/guides/skills/).

## Members

| Member | Description |
| --- | --- |
| [`new AgentBuilder()`](#new-agentbuilder) | A builder for `model`, such as `'anthropic/claude-sonnet-5-5'`: `<org>/<name>`, built through the provider set for `<org>`. |
| [`instruction()`](#instruction) | Set the system instruction: the system message the agent's history starts with, which shapes how the model works. |
| [`tool()`](#tool) | Give the agent one tool, by its desc. |
| [`tools()`](#tools) | Give the agent several tools at once, such as the descs `registerMcpStdio` hands back. |
| [`systemTools()`](#systemtools) | Give the agent the local-execution tools that suit its model's family: `shell`; for `openai`, `apply_patch`; for every other org, `read` (`read_file` for `google`), `write` and `edit`; and `imgread`, except for `deepseek-ai`, `moonshotai`, `zai-org` and `Qwen`, whose backends take tool results as text alone. |
| [`shellTool()`](#shelltool) | Give the agent only the `shell` tool, without the rest of `system_tools()`. |
| [`webSearchTool()`](#websearchtool) | Give the agent the `web_search` tool, which searches the web through the engines named, or through every engine when none are. |
| [`webFetchTool()`](#webfetchtool) | Give the agent the `web_fetch` tool, which isn't part of `system_tools()`. |
| [`history()`](#history) | Start the agent with this conversation, for example to resume an earlier session. |
| [`console()`](#console) | Run the agent's console tools in the session of `console`, a `@brekkylab/virtx` `ConsoleClient`, which the agent then shares. |
| [`browser()`](#browser) | The person's own Chromium-based browser with remote debugging on (Chrome by default: 144 or later, at `chrome://inspect/#remote-debugging`), which the agent drives with its `browser_*` tools over CDP from this process rather than the console, once the person allows it: see `BrowserOptions`. |
| [`memory()`](#memory) | Remember into the memory store at `memfile`, a path in the console. |
| [`skill()`](#skill) | Give the agent the skill in `dir`, a directory in the console holding a `SKILL.md`. |
| [`contextManager()`](#contextmanager) | Set how the agent keeps its history within the model's context: truncating old tool results once the input grows past a limit, and capping the images tool results hold. |
| [`thinkingEffort()`](#thinkingeffort) | How much the model thinks before it answers. |
| [`outputSchema()`](#outputschema) | Constrain replies to the JSON schema `schema`. |
| [`build()`](#build) | Build the agent: the model through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` read through the console. |

## Constructor

### `new AgentBuilder()`

A builder for `model`, such as `'anthropic/claude-sonnet-5-5'`: `<org>/<name>`, built
through the provider set for `<org>`.

```ts
new AgentBuilder(model: string)
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `string` | required | The model's desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI's default model. See [Model providers](../model-providers/). |

## Methods

### `instruction()`

Set the system instruction: the system message the agent's history starts with, which shapes how the model works. A leading system message in `history()` replaces it.

```ts
instruction(instruction: string): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `instruction` | `string` | required | The system prompt. |

**Returns** this builder, for chaining.

### `tool()`

Give the agent one tool, by its desc. The model sees the desc; `build()` builds the tool through the provider set for `desc.name`.

```ts
tool(desc: ToolDesc): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc) | required | The tool's name, description and parameter schema, such as what `registerTool` hands back. |

**Returns** this builder, for chaining.

### `tools()`

Give the agent several tools at once, such as the descs `registerMcpStdio` hands back.

```ts
tools(descs: ToolDesc[]): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `descs` | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]` | required | The tools' descs, each built through the provider set for its name. |

**Returns** this builder, for chaining.

### `systemTools()`

Give the agent the local-execution tools that suit its model's family: `shell`; for `openai`, `apply_patch`; for every other org, `read` (`read_file` for `google`), `write` and `edit`; and `imgread`, except for `deepseek-ai`, `moonshotai`, `zai-org` and `Qwen`, whose backends take tool results as text alone. These tools run in the console, so the agent needs `console()` too.

```ts
systemTools(): this
```

**Returns** this builder, for chaining.

### `shellTool()`

Give the agent only the `shell` tool, without the rest of `system_tools()`. It runs in the console, so the agent needs `console()` too.

```ts
shellTool(): this
```

**Returns** this builder, for chaining.

### `webSearchTool()`

Give the agent the `web_search` tool, which searches the web through the engines named, or through every engine when none are.

```ts
webSearchTool(engines?: WebSearchEngine[]): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `engines` (optional) | [`WebSearchEngine`](/docs/reference/nodejs/tools/#websearchengine)`[]` |  | The engines to search with. All of them when empty or left out. |

**Returns** this builder, for chaining.

### `webFetchTool()`

Give the agent the `web_fetch` tool, which isn't part of `system_tools()`. It fetches one URL per call, pages long bodies through `offset`, and allows one request per second per host.

```ts
webFetchTool(): this
```

**Returns** this builder, for chaining.

### `history()`

Start the agent with this conversation, for example to resume an earlier session. A leading system message here replaces the one `instruction()` would make; otherwise the instruction is put at the front of it.

```ts
history(history: Message[]): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `history` | [`Message`](/docs/reference/nodejs/messages/#message)`[]` | required | The earlier messages, oldest first, such as a previous agent's `history`. |

**Returns** this builder, for chaining.

### `console()`

Run the agent's console tools in the session of `console`, a `@brekkylab/virtx`
`ConsoleClient`, which the agent then shares.

The console needn't be started: the first batch of tool calls with a console tool starts it. Nothing makes a console for you. Without one, tools that don't need it still run, and console tools fail with an error.

```ts
console(console: ConsoleClient): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `console` | `ConsoleClient` | required | A `ConsoleClient` from `@brekkylab/virtx`. |

**Returns** this builder, for chaining.

### `browser()`

The person's own Chromium-based browser with remote debugging on (Chrome by default: 144 or
later, at `chrome://inspect/#remote-debugging`), which the agent drives with its
`browser_*` tools over CDP from this process rather than the console, once the person
allows it: see `BrowserOptions`.

Nothing starts until the agent first uses the browser, or `agent.startBrowser()`.

```ts
browser(options?: BrowserOptions | null): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `options` (optional) | [`BrowserOptions`](/docs/reference/nodejs/specs-and-options/#browseroptions)` \| null` |  | How to reach the browser. Left out, the person's own browser, with `careful` on. |

**Returns** this builder, for chaining.

### `memory()`

Remember into the memory store at `memfile`, a path in the console.

Adds the `mem_search` and `mem_insert` tools. The store must already exist (`mem init`): a missing file is reported by the first memory tool call, not by `build()`. Memory tools run in the console, so the agent needs `console()` too.

```ts
memory(memfile: string): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `memfile` | `string` | required | The memory store's path in the console. |

**Returns** this builder, for chaining.

### `skill()`

Give the agent the skill in `dir`, a directory in the console holding a `SKILL.md`. At `build()`, its frontmatter `name` and `description` are read through the console and listed in the system message with the skill's location. The agent reads the rest itself when it needs it, so it needs a console and a file-reading tool. Skills are only added to the system message `instruction()` makes, never to one already in `history()`. Call it once per skill.

```ts
skill(dir: string): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dir` | `string` | required | The skill's directory, in the console. |

**Returns** this builder, for chaining.

### `contextManager()`

Set how the agent keeps its history within the model's context: truncating old tool results once the input grows past a limit, and capping the images tool results hold. Settings left out keep their defaults. Without this call the defaults apply anyway: no truncation, and the image cap.

```ts
contextManager(options?: ContextManagerOptions | null): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `options` (optional) | [`ContextManagerOptions`](/docs/reference/nodejs/specs-and-options/#contextmanageroptions)` \| null` |  | The settings. Any left out keep their defaults. |

**Returns** this builder, for chaining.

### `thinkingEffort()`

How much the model thinks before it answers.

Left unset, the model or backend default applies.

```ts
thinkingEffort(effort: "low" | "medium" | "high"): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `effort` | `"low" \| "medium" \| "high"` | required | `"low"`, `"medium"` or `"high"`. |

**Returns** this builder, for chaining.

### `outputSchema()`

Constrain replies to the JSON schema `schema`.

```ts
outputSchema(schema: Record<string, unknown>): this
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `schema` | `Record<string, unknown>` | required | A JSON schema object. |

**Returns** this builder, for chaining.

### `build()`

Build the agent: the model through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` read through the console. Awaitable, since building the model and reading skills take time.

```ts
build(): Promise<Agent>
```

**Returns** `Promise<`[`Agent`](/docs/reference/nodejs/agent/)`>`: The agent.

**Throws** `AILOY_ERROR` when the model's org or a tool has no provider, a skill is given without a console or its frontmatter can't be read, or the browser settings don't fit together. The console's code, such as `NOT_FOUND`, when a skill's `SKILL.md` can't be read. `INVALID_ARG` when the builder was already built.
