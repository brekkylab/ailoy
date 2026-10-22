# Context manager

The agent sends its whole history to the model on every call, so a long run sends more with each step: every tool call adds its result, a file read, a page fetched or a command's output. The context manager keeps it down by replacing old tool results with a placeholder once the model's input grows past a limit.

## Turning it on

It is off until you set it, and both of its settings have defaults:

| Setting | Default | What it is |
| --- | --- | --- |
| `max_input_tokens` (`maxInputTokens`) | 30,000 | The input the model reports, in tokens, past which old tool results are replaced. |
| `preserve_recent_turns` (`preserveRecentTurns`) | 3 | How many of the latest user turns are left as they are. |

::: code-group

```python [Python]
agent = await (
    AgentBuilder("anthropic/claude-haiku-4-5")
    .system_tools()
    .console(console)
    .context_manager(max_input_tokens=50_000, preserve_recent_turns=2)
    .build()
)
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-4-5')
  .systemTools()
  .console(console_)
  .contextManager({ maxInputTokens: 50000, preserveRecentTurns: 2 })
  .build()
```

```rust [Rust]
use ailoy::agent::ContextManager;

let agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
    .system_tools()
    .console(console)
    .context_manager(ContextManager { max_input_tokens: 50_000, preserve_recent_turns: 2 })
    .build()
    .await?;
```

:::

`context_manager()` with no arguments (`contextManager()`, `ContextManager::default()`) uses both defaults.

## What it does

Before each call to the model, the agent looks at the last input token count the model reported. When that is more than `max_input_tokens`, it truncates the history:

- A turn is a user message and everything after it up to the next user message. The last `preserve_recent_turns` turns are left as they are, and with fewer turns than that in the history nothing changes.
- In the turns before those, each tool result's contents become the text `[context truncated]`. The result keeps the id of the call it answers, since APIs such as Anthropic's refuse a tool call that has no result.
- The system message, the user's messages and the model's own messages, tool calls included, stay whole. No message is removed.

The placeholders replace the results in the agent's history, so `agent.history` shows them too, and they stay in it.

Because the count comes from a call already made, the first call that goes past the limit is sent whole, and the history is truncated before the next one. A reply without `usage` leaves the last count in place, so a model that never reports usage, such as a model of your own whose replies leave it out, never triggers it. And since only tool results are replaced, a history whose messages alone are past the limit keeps growing.

::: warning
For `anthropic` models on the `api` backend, the count leaves out the input read from or written to the prompt cache, which Ailoy turns on for Claude. Once the cache holds the history, the count covers only what comes after it, often a handful of tokens, so it stays far below a limit the real input passes, and the context manager does not truncate.
:::

::: warning
With `preserve_recent_turns` at `0`, every tool result is replaced, those of the turn in progress too, so past the limit the model no longer sees the results of the tools it has just called.
:::
