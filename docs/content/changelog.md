---
title: Changelog
description: "What changed in each Ailoy release, newest first."
tableOfContents:
  maxHeadingLevel: 2
---

What changed in each release of Ailoy, newest first. Each release's full notes and the PRs behind them are on [GitHub releases](https://github.com/brekkylab/ailoy/releases).

## Unreleased

On `main`, not yet published to PyPI, npm or crates.io.

:::caution[Breaking changes]
- **The agent runs in a sandbox you hand it.** `ConsoleClient` and its mounts are gone. Make a virtx sandbox, mount what it needs where you make it, and pass it with `AgentBuilder.sandbox(...)`. The agent never starts, stops or removes it. (#572)
- **`vllm` is now `self-hosted`**, read from `SELF_HOSTED_BASE_URL` (and `SELF_HOSTED_API_KEY` if the server wants one). (#556)
- **The old memory tools are removed**: `AgentBuilder.memory` as it was, `mem_search` and `mem_insert`. The new memory below replaces them. (#581)
- **Python 3.11 or later and Node.js 22 or later**, as virtx 0.2 needs. (#600)
- **Sandboxes have no snapshots.** A sandbox's files last until it is removed; keep what a later run needs on a volume. (#600)
:::

### New features

- **Sandboxes you own.** An agent runs its tools in a virtx sandbox you pass it. The virtx server stops the machine when nothing uses it and boots it again when something does, and the tools of one batch now run side by side. See [Giving an agent a sandbox](/docs/guides/sandbox/). (#572)
- **Memory between sessions.** `memory(dir)` gives the agent a folder whose `MEMORY.md` holds short notes on the user. The notes are in the system message from the first turn, and the agent edits them with a `memory` tool. No sandbox is needed. (#592)
- **Your own model server.** The `self-hosted` backend runs a model on any server that speaks the OpenAI-compatible Chat Completions API: vLLM, llama-server, Ollama, LM Studio, SGLang, or a gateway such as LiteLLM. Each org can point at a server of its own, and a `models` map translates names like `Llama-3.3-70B-Instruct` to the server's `llama3.3:70b`. See [Model providers](/docs/guides/model-providers/). (#556)
- **EXAONE.** `LGAI-EXAONE/k-exaone_v2.5` runs on its own API with `EXAONE_API_KEY` alone. (#556)
- **A sandbox's programs as skills.** Each program virtx puts in a sandbox at `/abin` comes with a skill, and the agent lists those skills beside the ones you give it. (#581)

### Improvements

- Ailoy takes virtx 0.2.1 from PyPI, npm and crates.io. (#600, #616)
- The larger examples moved to [brekkylab/ailoy-examples](https://github.com/brekkylab/ailoy-examples). (#609)
- The docs have a Welcome page, section tabs, a grouped sidebar, a copy-as-Markdown action on each page, and shorter, rechecked guides. (#589, #599)
- Ailoy no longer depends on `jsonschema`. (#591)

## v0.3.2

*November 6, 2026*

### New features

- **Large images are shrunk.** An image past 2576 pixels on its longer edge is shrunk to that edge before it enters the history, so a screenshot agent stays within what a model accepts. (#545)
- **An image cap.** Past `max_images` (30) images in tool results, the oldest are replaced with a short note until `preserve_recent_images` (10) remain. Images a person attached are never replaced, and `max_images=0` turns the cap off. (#545)

  ```python
  AgentBuilder(model).context_manager(max_images=20, preserve_recent_images=5)
  ```

### Behaviour changes

- Every agent caps its images, with or without a context manager. The token limit stays opt-in. (#545)
- The history is truncated only as a turn begins: a turn runs to its end on the history it started with. (#561)

### Fixes

- A `claude_cli` reply reports its last request's usage, not the sum of every request in its run. (#561)
- A `gemini_cli` reply reports only its own request's usage, and counts thinking as output, as `gemini_api` does. (#563)
- The package tests pass on macOS and Windows again. (#540)

### Docs

- The docs moved to [ailoy.co/docs](https://ailoy.co/docs/). (#559)
- New guides: [Tools](/docs/guides/tools/), [Skills](/docs/guides/skills/) and [Context manager](/docs/guides/context-manager/), each with Python, Node.js and Rust examples. (#522)
- The new Ailoy logo in the README, the docs and the favicon. (#558)

## v0.3.1

*October 23, 2026*

:::caution
v0.3.1 is not compatible with v0.3.0, and there is no migration path.
:::

### New features

- **Models by their maker's name.** A model is `<org>/<name>`: `anthropic/claude-haiku-5-5`, `openai/gpt-5.6-luna`, `deepseek-ai/DeepSeek-V3.2`. The name stays the same whichever backend runs it.
- **No API key for Claude, GPT or Gemini.** By default they run through the CLI already on the machine (Claude Code, Codex or Gemini CLI) on your existing login. `set_lang_model_provider(org, backend)` switches an org to `api`, `bedrock`, `openrouter` or `vllm`. (#486)
- **Browser use.** `.browser()` gives the agent your own browser in a window of its own; `.browser(isolated=True)` gives it a separate one. It can navigate, read pages, click, type, scroll, take screenshots, and handle dialogs and tabs. (#510)
- **Tools registered like models.** `set_tool_provider(name, provider)` maps a tool name to how it is built, and every built-in tool is registered from the start. (#498)
- **Thinking and structured output everywhere.** `.thinking_effort(...)` and `.output_schema({...})` work on every backend.
- **Context manager.** `.context_manager(max_input_tokens, preserve_recent_turns)` blanks old tool results once the input passes the limit, counting cached input. (#534)
- **A skill for your coding agent.** `claude plugin marketplace add brekkylab/ailoy && claude plugin install ailoy@brekkylab` teaches it to build agents with Ailoy. (#500)

### Improvements

- `web_fetch` and `shell` size their results to fit in one tool result. (#531)
- A tool result the model's API refuses, an oversized image say, goes back to the model as an error and the conversation goes on. (#529)
- A reply cut off at the model's output limit is delivered, then reported as an error. (#523)
- `usage.input_tokens` counts everything a call read, cached or not.

### Removed

- The subagent and A2A APIs. (#496)
- Claude 4.5 models (Opus, Sonnet, Haiku) are not supported.

## v0.3.0

*October 2, 2026*

Ailoy was rebuilt as an agent builder that gives every agent a computer of its own: a microVM powered by [virtx](https://github.com/brekkylab/virtx).

- **Not compatible with v0.2.x**, and there is no migration path.
- **Rust.** Ailoy is now also a Rust crate, beside Python and Node.js.
- **New npm package:** `@brekkylab/ailoy`. `ailoy-node` will be deprecated, and `ailoy-web` is discontinued.
- **Removed:** local model inference, the knowledge and vector store (RAG), and the WebAssembly build.

## Earlier releases

v0.2.x and earlier were a different library, with local model inference, RAG and a WebAssembly build. Their notes are on GitHub:

| Version | Date | Highlights |
| --- | --- | --- |
| [v0.2.5](https://github.com/brekkylab/ailoy/releases/tag/v0.2.5) | January 9, 2026 | Example fixes, Korean docs |
| [v0.2.4](https://github.com/brekkylab/ailoy/releases/tag/v0.2.4) | January 6, 2026 | TVM moved to Rust; `InferenceConfig` renamed `LangModelInferConfig` |
| [v0.2.3](https://github.com/brekkylab/ailoy/releases/tag/v0.2.3) | December 18, 2025 | Model management APIs, `proxy_url` for `web_fetch` |
| [v0.2.2](https://github.com/brekkylab/ailoy/releases/tag/v0.2.2) | December 2, 2025 | Built-in `web_search_duckduckgo` and `web_fetch` tools |
| [v0.2.1](https://github.com/brekkylab/ailoy/releases/tag/v0.2.1) | November 20, 2025 | TVM update; `MessageDelta.to_message()` renamed `finish()` |
| [v0.2.0](https://github.com/brekkylab/ailoy/releases/tag/v0.2.0) | November 10, 2025 | Core rewritten in Rust |
| [v0.1.0](https://github.com/brekkylab/ailoy/releases/tag/v0.1.0) | August 14, 2025 | WebAssembly support |
| [v0.0.1 – v0.0.6](https://github.com/brekkylab/ailoy/releases?q=v0.0) | May – August 2025 | First releases |
