// Agents that drive a language model through tool-augmented turns.
//
// An `Agent` is built with an `AgentBuilder` and awaited; a turn is `agent.run(query)`,
// iterated with `for await`. The model and the tools are resolved by name from process-wide
// registries — `registerLangModel`, `registerTool`, `registerMcpStdio` and the rest — each
// of which starts with a `'default'` entry: the models whose API keys are in the
// environment, and every built-in tool.
//
// Messages, specs and tool descriptions are the objects their JSON form is; `types.d.ts`
// spells out their shapes.
//
// Where the tools run commands is a cortex console — `Console`, `Directory`, `Image` and the
// rest, exported from here too. They are the `cortex-node` classes built into this addon, not
// the same types as that package's: a `Console` built by `cortex-node` cannot be handed to an
// `AgentBuilder`, so build the console from here.
//
// `binding.js` is the loader `napi build` generates. What this file adds is what napi cannot
// declare from Rust: that a turn is an async iterable.

'use strict'

const binding = require('./binding.js')

binding.AgentRun.prototype[Symbol.asyncIterator] = function () {
  return this
}

module.exports = binding
