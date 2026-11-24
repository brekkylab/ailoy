---
title: "Model providers"
description: "How a model desc is built: the backend set for each org, or a model of your own."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

An agent or a `LangModel` names its model by a desc, `<org>/<name>`, or `<org>` alone for a CLI's default model. The provider set for the org decides how the model runs: the org's CLI, its own API, Bedrock, OpenRouter, vLLM, or a factory of your own. The same desc runs through every backend, except that DeepSeek's own API names its models without a version, so it takes DeepSeek's own names (`deepseek-ai/deepseek-chat`).

| Models | Naming | Desc |
| --- | --- | --- |
| Claude | `anthropic/<official API id>` | `anthropic/claude-sonnet-5-5` |
| GPT | `openai/<official API id>` | `openai/gpt-6.1-sol` |
| Gemini | `google/<official API id>` | `google/gemini-3.1-pro` |
| DeepSeek | Hugging Face repo id; DeepSeek's own name on its own API | `deepseek-ai/DeepSeek-V3.2`, `deepseek-ai/deepseek-chat` |
| Kimi | Hugging Face repo id | `moonshotai/Kimi-K2-Thinking` |
| GLM | Hugging Face repo id | `zai-org/GLM-4.6` |
| Qwen | Hugging Face repo id | `Qwen/Qwen3-235B-A22B-Instruct-2507` |

Each org starts with a provider. The default is the CLI where there is one (Claude Code for `anthropic`, Codex for `openai`, Gemini CLI for `google`), otherwise the org's own API.

| Org | CLI | APIs |
| --- | --- | --- |
| `anthropic` | Claude Code | own API, Bedrock, OpenRouter |
| `openai` | Codex | own API, Bedrock, OpenRouter |
| `google` | Gemini CLI | own API, OpenRouter |
| `deepseek-ai` | none | own API, Bedrock, OpenRouter, vLLM |
| `moonshotai` | none | own API, Bedrock, OpenRouter, vLLM |
| `zai-org` | none | own API, Bedrock, OpenRouter, vLLM |
| `Qwen` | none | own API, Bedrock, OpenRouter, vLLM |

The own APIs read their key from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY`, `ZAI_API_KEY` and `DASHSCOPE_API_KEY`. Bedrock reads `AWS_BEARER_TOKEN_BEDROCK` (and `AWS_REGION`), OpenRouter reads `OPENROUTER_API_KEY`, and vLLM uses the server at `VLLM_BASE_URL` (by default `http://localhost:8000/v1`), with `VLLM_API_KEY` if it's set. Every backend but the CLIs needs a name in the desc.

Providers are process-wide: setting one changes how every agent and `LangModel` built afterwards runs that org's models.

## Example

```js
const { LangModel, getLangModelProvider, setLangModelProvider } = require('@brekkylab/ailoy')

// Claude through its own API (reads ANTHROPIC_API_KEY) instead of Claude Code
setLangModelProvider('anthropic', 'api')
console.log(getLangModelProvider('anthropic/claude-sonnet-5-5')) // api

// Qwen on your own vLLM server (VLLM_BASE_URL)
setLangModelProvider('Qwen', 'vllm')

// A model of your own: here, one that repeats the last message back
/** @type {import('@brekkylab/ailoy').CustomLangModel} */
const echo = {
  infer(messages) {
    const last = messages[messages.length - 1].contents
    return { message: { role: 'assistant', contents: last }, finish_reason: { type: 'stop' } }
  },
}
setLangModelProvider('echo', () => echo)

async function main() {
  const model = await LangModel.create('echo/any-name')
  const out = await model.infer([{ role: 'user', contents: [{ type: 'text', text: 'hi' }] }])
  console.log(out.message.contents)
}

main()
```

Guides: [Model providers](/docs/guides/model-providers/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`setLangModelProvider()`](#setlangmodelprovider) | function | Build the models of `org` (`'anthropic'`, `'openai'`, `'google'`, `'deepseek-ai'`, …) through `provider`: a backend, or a factory, as the module says. |
| [`getLangModelProvider()`](#getlangmodelprovider) | function | How the model `desc` names (`'<org>/<name>'`, or `'<org>'`) is built: a backend's name, `'custom'` for a factory, or `null` when its org has no provider. |
| [`removeLangModelProvider()`](#removelangmodelprovider) | function | Stop building the models of `org`. |
| [`Backend`](#backend) | type | How a model is run. |
| [`CustomLangModel`](#customlangmodel) | interface | A model of your own, as a factory given to `setLangModelProvider` returns it. |

## `setLangModelProvider()`

Build the models of `org` (`'anthropic'`, `'openai'`, `'google'`, `'deepseek-ai'`, …)
through `provider`: a backend, or a factory, as the module says. Replaces the provider the
org had.

```ts
function setLangModelProvider(
  org: string,
  provider: "cli" | "api" | "bedrock" | "openrouter" | "vllm" | ((org: string, name: string | null) => LangModel | CustomLangModel),
): void
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `org` | `string` | required | The org, the part of a desc before the `/`. |
| `provider` | `"cli" \| "api" \| "bedrock" \| "openrouter" \| "vllm" \| ((org: string, name: string \| null) => `[`LangModel`](/docs/reference/nodejs/lang-model/)` \| `[`CustomLangModel`](/docs/reference/nodejs/model-providers/#customlangmodel)`)` | required | A backend's name (see [`Backend`](#backend)), or a factory called with the org and the name after it (`null` when the desc names only the org) that returns a `LangModel` or a [`CustomLangModel`](#customlangmodel). |

## `getLangModelProvider()`

How the model `desc` names (`'<org>/<name>'`, or `'<org>'`) is built: a backend's name,
`'custom'` for a factory, or `null` when its org has no provider.

```ts
function getLangModelProvider(
  desc: string,
): "cli" | "api" | "bedrock" | "openrouter" | "vllm" | "custom" | null
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | `string` | required | A desc, or just its org. |

**Returns** `"cli" | "api" | "bedrock" | "openrouter" | "vllm" | "custom" | null`: The backend's name, `'custom'` for a factory, or `null`.

## `removeLangModelProvider()`

Stop building the models of `org`. Answers whether it had a provider.

Models of the org then fail to build until a provider is set again.

```ts
function removeLangModelProvider(org: string): boolean
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `org` | `string` | required | The org. |

**Returns** `boolean`: Whether the org had a provider.

## `Backend`

How a model is run.

`"cli"` is the org's CLI (Claude Code, Codex or Gemini CLI); `"api"` the org's own API; `"bedrock"`, `"openrouter"` and `"vllm"` those services. Which backends an org has is in the tables above.

```ts
type Backend = "cli" | "api" | "bedrock" | "openrouter" | "vllm";
```

## `CustomLangModel`

A model of your own, as a factory given to `setLangModelProvider` returns it. Its methods are
called with `this` as the model. Without `inferStream`, streaming gives `infer`'s output as
one delta.

Its methods are called with `this` as the model.

```ts
interface CustomLangModel {
  infer(
    messages: Message[],
    tools: ToolDesc[],
    options: LangModelOptions,
  ): MessageOutput | Promise<MessageOutput>;
  inferStream?(
    messages: Message[],
    tools: ToolDesc[],
    options: LangModelOptions,
  ): AsyncIterable<MessageDeltaOutput>;
}
```

### `infer()`

Run the model on `messages` to completion. May return the output or an awaitable of it.

```ts
infer(
  messages: Message[],
  tools: ToolDesc[],
  options: LangModelOptions,
): MessageOutput | Promise<MessageOutput>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | [`Message`](/docs/reference/nodejs/messages/#message)`[]` | required | The conversation to run the model on, oldest first. |
| `tools` | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]` | required | The tools the model may call. |
| `options` | [`LangModelOptions`](/docs/reference/nodejs/specs-and-options/#langmodeloptions) | required | The call's settings. |

**Returns** [`MessageOutput`](/docs/reference/nodejs/outputs/#messageoutput)` | Promise<`[`MessageOutput`](/docs/reference/nodejs/outputs/#messageoutput)`>`: The model's whole message.

### `inferStream()`

Run the model on `messages` and yield its output as deltas while it's generated. Optional.

```ts
inferStream?(
  messages: Message[],
  tools: ToolDesc[],
  options: LangModelOptions,
): AsyncIterable<MessageDeltaOutput>
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | [`Message`](/docs/reference/nodejs/messages/#message)`[]` | required | The conversation to run the model on, oldest first. |
| `tools` | [`ToolDesc`](/docs/reference/nodejs/tools/#tooldesc)`[]` | required | The tools the model may call. |
| `options` | [`LangModelOptions`](/docs/reference/nodejs/specs-and-options/#langmodeloptions) | required | The call's settings. |

**Returns** `AsyncIterable<`[`MessageDeltaOutput`](/docs/reference/nodejs/outputs/#messagedeltaoutput)`>`: The deltas.
