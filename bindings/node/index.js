// Agents that drive a language model through tool-augmented turns.
//
// An `Agent` is built with an `AgentBuilder` and awaited; a turn is `agent.run(query)`, iterated
// with `for await`. Models and tools resolve by name from process-wide registries
// (`registerLangModel`, `registerTool`, `registerMcpStdio`, ...), each starting with a
// `'default'` entry: the models whose API keys are in the environment, and every built-in tool.
//
// Messages, specs and tool descriptions are plain objects in their JSON form; `types.d.ts`
// spells out their shapes.
//
// Tools run commands in a console from `@brekkylab/virtx` (`ConsoleClient`, `Directory`,
// `Recipe`, ...), whose exports are re-exported here: a `ConsoleClient` from either is the same
// class, and `AgentBuilder.console` shares its session with the agent.
//
// `binding.js` is the loader `napi build` generates; this file adds what napi cannot declare
// from Rust: a turn is an async iterable.

'use strict'

const virtx = require('@brekkylab/virtx')
const binding = require('./binding.js')

binding.AgentRun.prototype[Symbol.asyncIterator] = function () {
  return this
}

module.exports = { ...virtx, ...binding }
