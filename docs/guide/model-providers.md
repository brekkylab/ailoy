# Model Providers

## Naming a model

Name the model as `<org>/<name>` when you build the agent. The name is the one its maker gives it: the official API id for Claude, GPT and Gemini, and the Hugging Face repo id for open-weight models. The same name works whichever way the model runs.

| Models | Org | Example |
| --- | --- | --- |
| Claude | `anthropic` | `anthropic/claude-haiku-5-5` |
| GPT | `openai` | `openai/gpt-5.6-luna` |
| Gemini | `google` | `google/gemini-3.1-pro` |
| DeepSeek | `deepseek-ai` | `deepseek-ai/DeepSeek-V3.2` |
| Kimi | `moonshotai` | `moonshotai/Kimi-K2-Thinking` |
| GLM | `zai-org` | `zai-org/GLM-4.6` |
| Qwen | `Qwen` | `Qwen/Qwen3-235B-A22B-Instruct-2507` |

::: code-group

```python [Python]
agent = await AgentBuilder("anthropic/claude-haiku-5-5").build()
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-5-5').build()
```

```rust [Rust]
let agent = AgentBuilder::new("anthropic/claude-haiku-5-5").build().await?;
```

:::

For a CLI, the org alone (`anthropic`) runs the CLI's default model.

## Backends

Each org's models run on one **backend**:

| Backend | Runs the model through | Needs |
| --- | --- | --- |
| `cli` | The org's CLI: Claude Code (`claude`), Codex (`codex`) or Gemini CLI (`gemini`) | The CLI on `PATH`, logged in |
| `api` | The org's own API | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY` (or `MOONSHOT_API_KEY`), `ZAI_API_KEY` or `DASHSCOPE_API_KEY` (Qwen) |
| `bedrock` | Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK`; the region from `AWS_REGION`, then `AWS_DEFAULT_REGION`, else `us-east-1` |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` |
| `vllm` | A vLLM server | `VLLM_BASE_URL` (default `http://localhost:8000/v1`), and `VLLM_API_KEY` if the server wants one |

By default, Claude, GPT and Gemini run through their CLIs, and the other orgs through their own APIs. Keys are read from the environment when a model is built, so set them before that.

Not every org runs everywhere:

| Org | `cli` | `api` | `bedrock` | `openrouter` | `vllm` |
| --- | :-: | :-: | :-: | :-: | :-: |
| `anthropic` | ✓ | ✓ | ✓ | ✓ | |
| `openai` | ✓ | ✓ | ✓ | ✓ | |
| `google` | ✓ | ✓ | | ✓ | |
| `deepseek-ai`, `moonshotai`, `zai-org`, `Qwen` | | ✓ | ✓ | ✓ | ✓ |

Bedrock and OpenRouter name models their own way. Ailoy looks your model up in their catalogs, so you still write `anthropic/claude-haiku-5-5`, and the catalog decides which Bedrock or OpenRouter id it runs as.

## Choosing a backend

Set an org's backend with `set_lang_model_provider` (`setLangModelProvider` in Node.js) before building an agent. It applies to every model of that org built afterwards.

::: code-group

```python [Python]
import ailoy

ailoy.set_lang_model_provider("anthropic", "api")
agent = await AgentBuilder("anthropic/claude-haiku-5-5").build()
```

```js [Node.js]
const { setLangModelProvider } = require('@brekkylab/ailoy')

setLangModelProvider('anthropic', 'api')
const agent = await new AgentBuilder('anthropic/claude-haiku-5-5').build()
```

```rust [Rust]
use ailoy::model::{Backend, set_lang_model_provider};

set_lang_model_provider("anthropic", Backend::Api);
let agent = AgentBuilder::new("anthropic/claude-haiku-5-5").build().await?;
```

:::

`remove_lang_model_provider` (`removeLangModelProvider`) takes the org's provider away, after which its models can't be built.

## Model options

Two options work on every backend:

| Option | Builder method | Description |
| --- | --- | --- |
| `thinking_effort` | `thinking_effort` (`thinkingEffort`) | How much the model thinks before it answers: `low`, `medium` or `high`. Unset, the model's default. |
| `output_schema` | `output_schema` (`outputSchema`) | A JSON schema the model's answers must match. |

::: code-group

```python [Python]
agent = await (
    AgentBuilder("openai/gpt-5.6-luna")
    .thinking_effort("high")
    .output_schema({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})
    .build()
)
```

```js [Node.js]
const agent = await new AgentBuilder('openai/gpt-5.6-luna')
  .thinkingEffort('high')
  .outputSchema({ type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] })
  .build()
```

```rust [Rust]
use ailoy::{model::ThinkingEffort, to_value};

let agent = AgentBuilder::new("openai/gpt-5.6-luna")
    .thinking_effort(ThinkingEffort::High)
    .output_schema(to_value!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}))
    .build()
    .await?;
```

:::

## Running a model without an agent

`LangModel` builds a model the way an agent does, and runs it on a conversation you give it: no tools are called and no history is kept. `infer` returns the whole reply as a `MessageOutput`, and `infer_stream` (`inferStream`) yields `MessageDeltaOutput`s as the model writes.

::: code-group

```python [Python]
from ailoy import LangModel

model = await LangModel.create("anthropic/claude-haiku-5-5")
messages = [{"role": "user", "contents": [{"type": "text", "text": "Hi"}]}]

output = await model.infer(messages)
async for delta in model.infer_stream(messages, options={"thinking_effort": "low"}):
    ...
```

```js [Node.js]
const { LangModel } = require('@brekkylab/ailoy')

const model = await LangModel.create('anthropic/claude-haiku-5-5')
const messages = [{ role: 'user', contents: [{ type: 'text', text: 'Hi' }] }]

const output = await model.infer(messages)
for await (const delta of model.inferStream(messages, [], { thinking_effort: 'low' })) {
  // ...
}
```

```rust [Rust]
use ailoy::{
    message::{Message, Part, Role},
    model::{LangModelOptions, create_lang_model},
};

let model = create_lang_model("anthropic/claude-haiku-5-5").await?;
let messages = [Message::new(Role::User).with_contents([Part::text("Hi")])];

let output = model.infer(&messages, &[], &LangModelOptions::default()).await?;
let mut deltas = model.infer_stream(&messages, &[], &LangModelOptions::default());
```

:::

Tools, if you pass them, are only described to the model: its tool calls come back in the reply for you to run.

## Bringing your own model

An org can also be served by your own code. Instead of a backend name, give `set_lang_model_provider` a **factory**: it's called with the org and the name after it (`None`/`null` for the org alone) and returns the model. The model has

- `infer(messages, tools, options)`, returning a `MessageOutput` (or, in Python and Node.js, an awaitable or promise of one), and optionally
- `infer_stream(messages, tools, options)` (`inferStream`), yielding `MessageDeltaOutput`s. Without it, streaming gives `infer`'s reply as one delta.

The factory may also return a `LangModel`, to serve the org with a model from another one.

::: code-group

```python [Python]
class Echo:
    def __init__(self, name):
        self.name = name

    async def infer(self, messages, tools, options):
        text = messages[-1]["contents"][0]["text"]
        return {
            "message": {"role": "assistant", "contents": [{"type": "text", "text": text}]},
            "finish_reason": {"type": "stop"},
        }

ailoy.set_lang_model_provider("mine", lambda org, name: Echo(name))
agent = await AgentBuilder("mine/echo").build()
```

```js [Node.js]
class Echo {
  constructor(name) {
    this.name = name
  }
  async infer(messages, tools, options) {
    const text = messages[messages.length - 1].contents[0].text
    return {
      message: { role: 'assistant', contents: [{ type: 'text', text }] },
      finish_reason: { type: 'stop' },
    }
  }
}

setLangModelProvider('mine', (org, name) => new Echo(name))
const agent = await new AgentBuilder('mine/echo').build()
```

```rust [Rust]
use std::sync::Arc;

use ailoy::{
    message::{FinishReason, Message, MessageDeltaOutput, MessageOutput, Role},
    model::{InferLangModel, LangModelOptions, LangModelProvider, set_lang_model_provider},
    tool::ToolDesc,
};
use futures::{future::BoxFuture, stream::{self, BoxStream}, StreamExt as _};

struct Echo;

impl InferLangModel for Echo {
    fn infer(
        &self,
        messages: &[Message],
        _tools: &[ToolDesc],
        _options: &LangModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
        let contents = messages.last().map(|m| m.contents.clone()).unwrap_or_default();
        Box::pin(async move {
            Ok(MessageOutput {
                message: Message::new(Role::Assistant).with_contents(contents),
                finish_reason: FinishReason::Stop {},
                usage: None,
            })
        })
    }

    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let output = self.infer(messages, tools, options);
        stream::once(async move { output.await.map(MessageDeltaOutput::from) }).boxed()
    }
}

set_lang_model_provider("mine", LangModelProvider::custom(|_org, _name| Ok(Arc::new(Echo))));
let agent = AgentBuilder::new("mine/echo").build().await?;
```

:::

In Python and Node.js the factory itself must return the model directly, not an awaitable or promise of it. The model's methods are called with the messages, tools and options as dicts or objects, the same shapes the [message format](./message-format) describes.
