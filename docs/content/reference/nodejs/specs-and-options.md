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

[`AgentSpec`](#agentspec) is what makes an agent distinct (its model, instruction, tools, model options, search engines and skills) as one object, for `Agent.fromSpec`. [`LangModelOptions`](#langmodeloptions) are the settings every model call takes, as `LangModel.infer()` and a custom model receive them. `BrowserOptions` and `ContextManagerOptions` are the objects `AgentBuilder.browser()` and `AgentBuilder.contextManager()` take.

## Example

```js
const { Agent, AgentBuilder } = require('@brekkylab/ailoy')

// An agent as data: store it as JSON, load it, build it
const spec = {
  model: 'anthropic/claude-sonnet-5-5',
  instruction: 'Be brief.',
  model_options: { thinking_effort: 'low' },
}
const saved = JSON.stringify(spec)

async function main() {
  const agent = await Agent.fromSpec(JSON.parse(saved))
  await agent.close()

  // The options objects the builder takes
  const browsing = await new AgentBuilder('anthropic/claude-sonnet-5-5')
    .browser({ isolated: true, headless: true })
    .contextManager({ maxInputTokens: 100_000, preserveRecentTurns: 2 })
    .build()
  await browsing.close()
}

main()
```

Guides: [Defining an agent](/docs/guides/agents/), [Context manager](/docs/guides/context-manager/), [Using a browser](/docs/guides/using-browser/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`AgentSpec`](#agentspec) | type | What makes an agent distinct, as data: its model, instruction and tools. |
| [`LangModelOptions`](#langmodeloptions) | type | The settings every model call takes, whichever backend runs it. |
| [`ThinkingEffort`](#thinkingeffort) | type | How much the model thinks before it answers. |
| [`BrowserOptions`](#browseroptions) | type | How `AgentBuilder.browser()` reaches a browser: the person's own, with remote debugging on, or with `isolated` one of the agent's own. |
| [`UserDataDir`](#userdatadir) | type | Any other Chromium-based browser, by the user data directory it writes `DevToolsActivePort` in. |
| [`ContextManagerOptions`](#contextmanageroptions) | type | The settings `AgentBuilder.contextManager()` takes. |

## `AgentSpec`

What makes an agent distinct, as data: its model, instruction and tools. The console, browser, memory and history aren't part of it.

```ts
interface AgentSpec {
  model: string;
  instruction?: string;
  tools?: ToolDesc[];
  model_options?: LangModelOptions;
  web_search_engines?: WebSearchEngine[];
  skills?: string[];
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | `string` | yes | The model's desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI's default model. See [Model providers](../model-providers/). |
| `instruction` | `string` | no | The system instruction. |
| `tools` | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]` | no | The tools' descs. Each name needs a provider set for it. |
| `model_options` | [`LangModelOptions`](/docs/reference/nodejs/specs-and-options/#langmodeloptions) | no | Thinking effort and output schema. |
| `web_search_engines` | [`WebSearchEngine`](/docs/reference/nodejs/tools/#websearchengine)`[]` | no | The engines `web_search` uses, when it's among `tools`. All of them when left out. |
| `skills` | `string[]` | no | Skill directories in the console, each holding a `SKILL.md`. |

## `LangModelOptions`

The settings every model call takes, whichever backend runs it.

```ts
interface LangModelOptions {
  thinking_effort?: ThinkingEffort;
  output_schema?: any;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `thinking_effort` | [`ThinkingEffort`](/docs/reference/nodejs/specs-and-options/#thinkingeffort) | no | How much the model thinks. Left out, the model or backend default. |
| `output_schema` | `any` | no | A JSON schema the output must match. |

## `ThinkingEffort`

How much the model thinks before it answers.

```ts
type ThinkingEffort = "low" | "medium" | "high";
```

## `BrowserOptions`

How `AgentBuilder.browser()` reaches a browser: the person's own, with remote debugging on, or with `isolated` one of the agent's own.

```ts
interface BrowserOptions {
  isolated?: boolean;
  headless?: boolean;
  executable?: string;
  profile?: string;
  userBrowser?: "chrome" | "chrome-beta" | "chrome-dev" | "chrome-canary" | "chromium" | "edge" | "brave" | { userDataDir: string };
  careful?: boolean;
  startUrl?: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `isolated` | `boolean` | no | A browser of the agent's own, apart from the user's: started on first use and closed with the agent. `false` by default: the user's own browser. |
| `headless` | `boolean` | no | For `isolated`: no window, as on a server. `false` by default. |
| `executable` | `string` | no | For `isolated`: the Chromium-based browser to start, by its executable; else one installed here (Chrome, Chromium, Edge or Brave). |
| `profile` | `string` | no | For `isolated`: the user data directory it keeps, so sign-ins last; a fresh one by default. |
| `userBrowser` | `"chrome" \| "chrome-beta" \| "chrome-dev" \| "chrome-canary" \| "chromium" \| "edge" \| "brave" \| { userDataDir: string }` | no | For the user's browser, which one: `'chrome'`, `'chrome-beta'`, `'chrome-dev'`, `'chrome-canary'`, `'chromium'`, `'edge'`, `'brave'`, or any other by `{ userDataDir }`; with none, whichever turned remote debugging on last. |
| `careful` | `boolean` | no | Whether the agent is told to use sites as a person would, so sites checking for bots are less likely to turn it away. `true` by default. |
| `startUrl` | `string` | no | The page the agent's tab opens on. |

## `UserDataDir`

Any other Chromium-based browser, by the user data directory it writes `DevToolsActivePort`
in.

```ts
interface UserDataDir {
  userDataDir: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `userDataDir` | `string` | yes | The browser's user data directory. |

## `ContextManagerOptions`

The settings `AgentBuilder.contextManager()` takes. Any left out keep their defaults.

```ts
interface ContextManagerOptions {
  maxInputTokens?: number;
  preserveRecentTurns?: number;
  maxImages?: number;
  preserveRecentImages?: number;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `maxInputTokens` | `number` | no | Truncate the history as a turn begins once the previous model call's input went past this many tokens. `0` turns truncation off. Truncation keeps the system message and the recent turns, and replaces each earlier tool result's contents with `"[context truncated]"`, keeping its `id`. Whole turns are never dropped. Default `0`. |
| `preserveRecentTurns` | `number` | no | How many recent user turns truncation leaves alone. The turn just begun is one of them, so `0` keeps it too. Default `3`. |
| `maxImages` | `number` | no | How many images tool results may hold in the history at the start of a turn. Past it, the oldest become `"[image from an earlier turn removed]"`. Images a person attached are left alone. `0` turns the cap off. Default `30`. |
| `preserveRecentImages` | `number` | no | How many tool-result images are left once `max_images` is passed. They're removed several at a time, so the prompt cache breaks once in several turns rather than every turn. Default `10`. |
