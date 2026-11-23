---
title: AgentBuilder
description: An AgentBuilder, filled in place and emptied by build().
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

An `AgentBuilder`, filled in place and emptied by `build()`.

In place rather than by value: the Rust builder is consumed by each call and is not
`Clone`, so there is exactly one of it to hand along.
Each method returns the same object so calls chain as they do in Rust.

## Constructors

### Constructor

```ts
new AgentBuilder(model): AgentBuilder;
```

A builder for `model`, such as `'anthropic/claude-sonnet-5-5'`: `<org>/<name>`, built
through the provider set for `<org>`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `model` | `string` |

#### Returns

`AgentBuilder`

## Methods

### instruction()

```ts
instruction(instruction): this;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `instruction` | `string` |

#### Returns

`this`

***

### tool()

```ts
tool(desc): this;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `desc` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/) |

#### Returns

`this`

***

### tools()

```ts
tools(descs): this;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `descs` | [`ToolDesc`](/docs/reference/nodejs/tool-desc/)[] |

#### Returns

`this`

***

### systemTools()

```ts
systemTools(): this;
```

#### Returns

`this`

***

### shellTool()

```ts
shellTool(): this;
```

#### Returns

`this`

***

### webSearchTool()

```ts
webSearchTool(engines?): this;
```

`engines` by name, `'Google'`, `'DuckDuckGo'`, …, or every engine when empty.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `engines?` | [`WebSearchEngine`](/docs/reference/nodejs/web-search-engine/)[] |

#### Returns

`this`

***

### webFetchTool()

```ts
webFetchTool(): this;
```

#### Returns

`this`

***

### history()

```ts
history(history): this;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `history` | [`Message`](/docs/reference/nodejs/message/)[] |

#### Returns

`this`

***

### console()

```ts
console(console): this;
```

Run the agent's console tools in the session of `console`, a `@brekkylab/virtx`
`ConsoleClient`, which the agent then shares.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `console` | `ConsoleClient` |

#### Returns

`this`

***

### browser()

```ts
browser(options?): this;
```

The person's own Chromium-based browser with remote debugging on (Chrome by default: 144 or
later, at `chrome://inspect/#remote-debugging`), which the agent drives with its
`browser_*` tools over CDP from this process rather than the console, once the person
allows it: see `BrowserOptions`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `options?` | [`BrowserOptions`](/docs/reference/nodejs/browser-options/) \| `null` |

#### Returns

`this`

***

### memory()

```ts
memory(memfile): this;
```

Remember into the memory store at `memfile`, a path in the console.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `memfile` | `string` |

#### Returns

`this`

***

### skill()

```ts
skill(dir): this;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `dir` | `string` |

#### Returns

`this`

***

### contextManager()

```ts
contextManager(options?): this;
```

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `options?` | [`ContextManagerOptions`](/docs/reference/nodejs/context-manager-options/) \| `null` |

#### Returns

`this`

***

### thinkingEffort()

```ts
thinkingEffort(effort): this;
```

How much the model thinks before it answers.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `effort` | `"low"` \| `"medium"` \| `"high"` |

#### Returns

`this`

***

### outputSchema()

```ts
outputSchema(schema): this;
```

Constrain replies to the JSON schema `schema`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `schema` | `Record`\<`string`, `unknown`\> |

#### Returns

`this`

***

### build()

```ts
build(): Promise<Agent>;
```

Make the agent, settling with it, a promise, as `AgentBuilder::build` is a future: it
builds the model and reads the skills through the console.

#### Returns

`Promise`\<[`Agent`](/docs/reference/nodejs/agent/)\>
