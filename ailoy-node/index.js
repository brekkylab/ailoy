// Agents that drive a language model through tool-augmented turns.
//
// An `Agent` is built with an `AgentBuilder` and awaited; a turn is `agent.run(query)`, iterated
// with `for await`. A model is named `<org>/<name>` (`'anthropic/claude-sonnet-5-5'`) and runs on
// the backend set for its org with `setLangModelProvider`: by default its CLI, or its own API
// when it has none. `LangModel.create(desc)` builds one to run without an agent. A tool is built
// by name through the provider set for it (`setToolProvider`, `registerTool`, `registerMcpStdio`,
// ...); every built-in tool has one from the start.
//
// Messages, specs and tool descriptions are plain objects in their JSON form; `types.d.ts`
// spells out their shapes.
//
// Tools run commands in a sandbox from `@brekkylab/virtx` (`Client`, `Sandbox`, ...), which
// ailoy depends on but does not re-export: make one with virtx and hand it to
// `AgentBuilder.sandbox`; it stays the caller's, and the agent runs in it.
//
// `binding.js` is the loader `napi build` generates; this file adds what napi cannot declare
// from Rust: a turn is an async iterable, and the addon opens the library virtx opened, by the
// path virtx's `libraryPath()` answers.

'use strict'

const binding = require('./binding.js')

binding._useVirtx(require('@brekkylab/virtx').libraryPath)
delete binding._useVirtx

binding.AgentRun.prototype[Symbol.asyncIterator] = function () {
  return this
}

module.exports = binding
