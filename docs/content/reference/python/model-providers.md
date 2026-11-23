---
title: "Model providers"
description: "How a model desc is built: the backend set for each org, or a model of your own."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

An agent or a `LangModel` names its model by a desc, `<org>/<name>`, or `<org>` alone for a CLI's default model. The provider set for the org decides how the model runs: the org's CLI, its own API, Bedrock, OpenRouter, vLLM, or a factory of your own. The same desc runs through every backend.

| Models | Naming | Desc |
| --- | --- | --- |
| Claude | `anthropic/<official API id>` | `anthropic/claude-sonnet-5-5` |
| GPT | `openai/<official API id>` | `openai/gpt-6.1-sol` |
| Gemini | `google/<official API id>` | `google/gemini-3.1-pro` |
| DeepSeek | Hugging Face repo id | `deepseek-ai/DeepSeek-V3.2` |
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

```python
import asyncio

from ailoy import LangModel, get_lang_model_provider, set_lang_model_provider
from ailoy.types import LangModelOptions, Message, MessageOutput, ToolDesc

# Claude through its own API (reads ANTHROPIC_API_KEY) instead of Claude Code
set_lang_model_provider("anthropic", "api")
print(get_lang_model_provider("anthropic/claude-sonnet-5-5"))  # api

# Qwen on your own vLLM server (VLLM_BASE_URL)
set_lang_model_provider("Qwen", "vllm")


# A model of your own: here, one that repeats the last message back
class Echo:
    def infer(
        self, messages: list[Message], tools: list[ToolDesc], options: LangModelOptions
    ) -> MessageOutput:
        last = messages[-1]["contents"]
        return {"message": {"role": "assistant", "contents": last}, "finish_reason": {"type": "stop"}}


set_lang_model_provider("echo", lambda org, name: Echo())


async def main() -> None:
    model = await LangModel.create("echo/any-name")
    out = await model.infer([{"role": "user", "contents": [{"type": "text", "text": "hi"}]}])
    print(out["message"]["contents"])


asyncio.run(main())
```

Guides: [Model providers](/docs/guides/model-providers/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`set_lang_model_provider()`](#set_lang_model_provider) | function | Build the models of `org` (`"anthropic"`, `"openai"`, `"google"`, `"deepseek-ai"`, …) through `provider`: a backend, `"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or `"vllm"`, or a factory, as the module says. |
| [`get_lang_model_provider()`](#get_lang_model_provider) | function | How the model `desc` names (`"<org>/<name>"`, or `"<org>"`) is built: a backend's name, `"custom"` for a factory, or `None` when its org has no provider. |
| [`remove_lang_model_provider()`](#remove_lang_model_provider) | function | Stop building the models of `org`. |
| [`Backend`](#backend) | type | How a model is run. |
| [`CustomLangModel`](#customlangmodel) | interface | A model of your own, as a factory given to `set_lang_model_provider` returns it. |
| [`StreamingCustomLangModel`](#streamingcustomlangmodel) | interface | A `CustomLangModel` that streams: one with `infer_stream` as well as `infer`. |
| [`LangModelFactory`](#langmodelfactory) | type | What `set_lang_model_provider` takes for a model of your own: called with the org and the name after it (`None` when the desc names only the org), it returns a `LangModel` or a `CustomLangModel`. |

## `set_lang_model_provider()`

Build the models of `org` (`"anthropic"`, `"openai"`, `"google"`, `"deepseek-ai"`, …)
through `provider`: a backend, `"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or `"vllm"`, or a factory, as the module says. Replaces the provider the org had.

```python
def set_lang_model_provider(
    org: str, provider: Backend | LangModelFactory
) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `org` | `str` | required | The org, the part of a desc before the `/`. |
| `provider` | [`Backend`](/docs/reference/python/model-providers/#backend)` \| `[`LangModelFactory`](/docs/reference/python/model-providers/#langmodelfactory) | required | A backend's name (see [`Backend`](#backend)), or a factory called with the org and the name after it (`None` when the desc names only the org) that returns a `LangModel` or a [`CustomLangModel`](#customlangmodel). |

## `get_lang_model_provider()`

How the model `desc` names (`"<org>/<name>"`, or `"<org>"`) is built: a backend's name,
`"custom"` for a factory, or `None` when its org has no provider.

```python
def get_lang_model_provider(
    desc: str,
) -> Backend | Literal["custom"] | None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `desc` | `str` | required | A desc, or just its org. |

**Returns** [`Backend`](/docs/reference/python/model-providers/#backend)` | Literal['custom'] | None`: The backend's name, `"custom"` for a factory, or `None`.

## `remove_lang_model_provider()`

Stop building the models of `org`. Answers whether it had a provider.

Models of the org then fail to build until a provider is set again.

```python
def remove_lang_model_provider(org: str) -> bool
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `org` | `str` | required | The org. |

**Returns** `bool`: Whether the org had a provider.

## `Backend`

How a model is run.

`"cli"` is the org's CLI (Claude Code, Codex or Gemini CLI); `"api"` the org's own API; `"bedrock"`, `"openrouter"` and `"vllm"` those services. Which backends an org has is in the tables above.

```python
Backend = Literal["cli", "api", "bedrock", "openrouter", "vllm"]
```

## `CustomLangModel`

A model of your own, as a factory given to `set_lang_model_provider` returns it.
`infer_stream` is optional: without it, streaming gives `infer`'s output as one delta.

Declared in the type stubs only, for type checking: import it under `TYPE_CHECKING` from `ailoy._ailoy`. Any object with these methods works.

### `infer()`

Run the model on `messages` to completion. May return the output or an awaitable of it.

```python
def infer(
    messages: list[Message], tools: list[ToolDesc], options: LangModelOptions
) -> MessageOutput | Awaitable[MessageOutput]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | `list[`[`Message`](/docs/reference/python/messages/#message)`]` | required | The conversation to run the model on, oldest first. |
| `tools` | `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`]` | required | The tools the model may call. |
| `options` | [`LangModelOptions`](/docs/reference/python/specs-and-options/#langmodeloptions) | required | The call's settings. |

**Returns** [`MessageOutput`](/docs/reference/python/outputs/#messageoutput)` | Awaitable[`[`MessageOutput`](/docs/reference/python/outputs/#messageoutput)`]`: The model's whole message.

## `StreamingCustomLangModel`

A `CustomLangModel` that streams: one with `infer_stream` as well as `infer`.

Declared in the type stubs only, for type checking.

### `infer_stream()`

Run the model on `messages` and yield its output as deltas while it's generated.

```python
def infer_stream(
    messages: list[Message], tools: list[ToolDesc], options: LangModelOptions
) -> AsyncIterable[MessageDeltaOutput]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | `list[`[`Message`](/docs/reference/python/messages/#message)`]` | required | The conversation to run the model on, oldest first. |
| `tools` | `list[`[`ToolDesc`](/docs/reference/python/tools/#tooldesc)`]` | required | The tools the model may call. |
| `options` | [`LangModelOptions`](/docs/reference/python/specs-and-options/#langmodeloptions) | required | The call's settings. |

**Returns** `AsyncIterable[`[`MessageDeltaOutput`](/docs/reference/python/outputs/#messagedeltaoutput)`]`: The deltas.

## `LangModelFactory`

What `set_lang_model_provider` takes for a model of your own: called with the org and the name after it (`None` when the desc names only the org), it returns a `LangModel` or a `CustomLangModel`. Declared in the type stubs only, for type checking.

```python
LangModelFactory = Callable[[str, str | None], LangModel | CustomLangModel]
```
