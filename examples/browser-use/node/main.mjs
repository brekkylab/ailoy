// An agent with a browser, waiting for what you ask of it.
//
//     node main.mjs [--host]
//
// The browser starts before the first request: Chromium in a VM of its own, which you watch and
// use in yours at the address printed, opened on the ailoy docs. Each line you type is the next
// turn; `/exit` ends. When a site needs you, to sign in say, the agent says so and ends its
// turn: do it in the viewer, then tell it to go on.
//
// With `--host` it uses your own Chrome instead, in a window of its own: turn on remote
// debugging at chrome://inspect/#remote-debugging (Chrome 144 or later) and allow Chrome's
// prompt, which appears as it starts.
//
// `AILOY_MODEL` is the agent's model, `anthropic/claude-sonnet-5` by default; its provider's API
// key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), in the environment or in `.env`.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'

const HERE = dirname(fileURLToPath(import.meta.url))

const { AgentBuilder, ConsoleClient, Recipe, ensureVirtx } = ailoy

// Where the browser starts.
const START = 'https://brekkylab.github.io/ailoy/'

// What to ask, for a start.
const SUGGESTIONS = [
  "Solve today's New York Times Wordle puzzle.",
  'Search Skyscanner and Google Flights for round-trip weekend flights from Seoul to Tokyo next month, and list the three cheapest by price.',
  'Find three wireless mice under $50 at an online store, and compare their prices and main features.',
]

// From the nearest `.env` up from this file. What the environment already has wins.
function loadDotenv() {
  for (let dir = HERE; ; dir = dirname(dir)) {
    const path = join(dir, '.env')
    if (existsSync(path)) return process.loadEnvFile(path)
    if (dirname(dir) === dir) return
  }
}

function show(message) {
  if (message.role !== 'assistant') return
  for (const part of message.contents) {
    if (part.type === 'text' && part.text.trim()) console.log(part.text.trim())
  }
  for (const call of message.tool_calls ?? []) {
    const { name, arguments: args } = call.function
    console.log(`  [${name}] ${JSON.stringify(args).slice(0, 160)}`)
  }
}

async function main(model, host) {
  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()
  // The browser is driven from this console over the network, so it has to have one.
  const console_ = await ConsoleClient.builder().image(new Recipe('debian:trixie-slim')).network(true).build()
  const agent = await new AgentBuilder(model)
    .systemTools()
    .console(console_)
    .browser(host ? { mode: 'host' } : {})
    .build()
  console.log(`model  ${model}\n`)
  try {
    if (host) console.log("Allow Chrome's prompt to let the agent in.")
    const url = await agent.startBrowser()
    console.log(url ? `Watch and use the browser at ${url}\n` : 'The agent has a window of its own in your Chrome.\n')

    // The app drives the agent's browser too, from the same console.
    console.log('Opening the ailoy docs as an example.\n')
    const opened = await console_.exec(['agent-browser', 'open', START])
    if (opened.code !== 0) {
      console.log(`Could not open ${START}: ${Buffer.concat([opened.stdout, opened.stderr]).toString().trim()}\n`)
    }

    console.log('Ask anything, or try:')
    for (const suggestion of SUGGESTIONS) console.log(`  - ${suggestion}`)
    console.log('`/exit` ends.\n')

    const prompt = () => process.stdout.write('> ')
    prompt()
    // Ends at `/exit` or once stdin closes.
    for await (const line of createInterface({ input: process.stdin })) {
      const query = line.trim()
      if (query === '/exit') break
      if (query) {
        for await (const { message } of agent.run(query)) show(message)
        console.log()
      }
      prompt()
    }
  } finally {
    await agent.close()
    await console_.close()
  }
}

loadDotenv()
await main(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5', process.argv.slice(2).includes('--host'))
