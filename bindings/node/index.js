// Agents that drive a language model through tool-augmented turns.
//
// An `Agent` is built with an `AgentBuilder` and awaited; a turn is `agent.run(query)`, iterated
// with `for await`. A model is named `<org>/<name>` (`'anthropic/claude-sonnet-5-5'`) and runs on
// the backend set for its org with `setLangModelProvider`: by default its CLI, or its own API
// when it has none. `LangModel.create(desc)` builds one to run without an agent. Tools resolve
// by name from process-wide registries (`registerTool`, `registerMcpStdio`, ...), each starting
// with a `'default'` entry of every built-in tool.
//
// Messages, specs and tool descriptions are plain objects in their JSON form; `types.d.ts`
// spells out their shapes.
//
// Tools run commands in a console from `@brekkylab/virtx` (`ConsoleClient`, `Recipe`, ...),
// which ailoy depends on but does not re-export: build one with virtx and hand it to
// `AgentBuilder.console`, which shares its session with the agent.
//
// `binding.js` is the loader `napi build` generates; this file adds what napi cannot declare
// from Rust: a turn is an async iterable.

'use strict'

const binding = require('./binding.js')

binding.AgentRun.prototype[Symbol.asyncIterator] = function () {
  return this
}

module.exports = binding
