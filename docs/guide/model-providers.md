# Model Providers

## Choosing a model

Name the model as `<provider>/<model id>` when you build the agent. The prefix picks the provider, and the rest is passed to that provider's API as the model id.

::: code-group

```python [Python]
agent = await AgentBuilder("anthropic/claude-haiku-4-5").build()
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-4-5').build()
```

```rust [Rust]
let agent = AgentBuilder::new("anthropic/claude-haiku-4-5").build().await?;
```

:::

A provider is available once its API key is set in the environment. Ailoy reads the keys the first time an agent is built, so set them before that:

```sh
export ANTHROPIC_API_KEY=...
```

Building an agent for a provider with no key set fails with `no entry for model '...' in lang_model_provider 'default'`.

## Providers

| Prefix | Provider | Environment variable | Example |
| --- | --- | --- | --- |
| `openai/` | OpenAI | `OPENAI_API_KEY` | `openai/gpt-5.6-luna` |
| `anthropic/` | Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-haiku-4-5` |
| `google/` | Google Gemini | `GEMINI_API_KEY` | `google/gemini-2.5-flash` |
| `x-ai/` | xAI Grok | `XAI_API_KEY` | `x-ai/grok-4-fast` |
| `deepseek/` | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| `moonshotai/` | Moonshot AI (Kimi) | `KIMI_API_KEY` | `moonshotai/kimi-k2-thinking` |
| `openrouter/` | OpenRouter | `OPENROUTER_API_KEY` | `openrouter/openai/gpt-5` |
| `bedrock/` | Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` | `bedrock/global.anthropic.claude-sonnet-5` |

- **OpenRouter** takes OpenRouter's own model ids, which include the vendor: `openrouter/openai/gpt-5` sends `openai/gpt-5`.
- **Bedrock** takes Bedrock model or inference-profile ids. The region comes from `AWS_REGION`, then `AWS_DEFAULT_REGION`, and defaults to `us-east-1`.
