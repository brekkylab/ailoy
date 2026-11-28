// Agents that drive a language model through tool-augmented turns.
//
// An `Agent` is made from a spec with `Agent.fromSpec` and awaited; a turn is `agent.run(query)`,
// iterated with `for await`. A model is named `<org>/<name>` (`'anthropic/claude-sonnet-5-5'`) and runs on
// the backend set for its org with `setLangModelProvider`: by default its CLI, or its own API
// when it has none. `LangModel.create(desc)` builds one to run without an agent. A tool is given
// by name and made through the provider of its name (`registerTool`, `registerMcpStdio`, ...);
// every built-in tool is provided from the start.
//
// Messages, specs and tool descriptions are plain objects in their JSON form; `types.d.ts`
// spells out their shapes.
//
// Tools run commands in a sandbox from `@brekkylab/virtx`, which ailoy depends on but does not
// re-export: each agent's sandbox is made, as `Agent.fromSpec`'s `sandbox` says, with the virtx
// `Client` given to `setVirtx`, and goes with the agent.
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
