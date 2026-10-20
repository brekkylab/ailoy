// An agent with a browser, waiting for what you ask of it.
//
//     node main.mjs [--noshell]
//
// The agent uses your own browser (Chrome by default), in a window of its own opened on the ailoy
// docs, and drives it with its `browser_*` tools. It also has a console with python and curl, for
// what is easier done there, such as fetching a page's data or computing over it; `--noshell`
// leaves it out, so the agent has the browser alone and needs no console server. Turn on remote
// debugging at chrome://inspect/#remote-debugging (Chrome 144 or later) and allow the browser's
// prompt, which appears as it starts. Each line you type is the next turn; `/exit` ends. When a
// site needs you, to sign in say, the agent says so and ends its turn: do it in its window, then
// tell it to go on.
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
import virtx from '@brekkylab/virtx'

const HERE = dirname(fileURLToPath(import.meta.url))

const { AgentBuilder } = ailoy
const { ConsoleClient, Recipe, ensureVirtx } = virtx

// Where the browser starts.
const START = 'https://brekkylab.github.io/ailoy/'

// What the agent has, with a console and without.
const WITH_CONSOLE =
  "a web browser and a console: the browser is the user's own, which you drive with the browser tools"
const BROWSER_ONLY = "a web browser: the user's own, which you drive with the browser tools"

// What the agent is for.
const INSTRUCTION = [
  'Do what the user asks, and say briefly what you are doing as you go.',
  'When you are done, report what you found plainly, with the figures and links that matter.',
  "Answer in the user's language.",
].join(' ')

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

async function main(model, shell) {
  // The browser tools reach the browser from this process; the console is for the other tools.
  const has = shell ? WITH_CONSOLE : BROWSER_ONLY
  let builder = new AgentBuilder(model)
    .instruction(`You are a helpful assistant with ${has}. ${INSTRUCTION}`)
    .browser({ startUrl: START })
  let console_ = null
  if (shell) {
    // The console server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none.
    await ensureVirtx()
    console_ = await ConsoleClient.builder()
      .image(
        new Recipe('python:3.12-slim-trixie').step(
          'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*',
        ),
      )
      .network(true)
      .build()
    builder = builder.systemTools().console(console_)
  }
  const agent = await builder.build()
  console.log(`model     ${model}`)
  console.log(`has       ${shell ? 'browser, console' : 'browser'}\n`)
  try {
    console.log("Allow the browser's prompt to let the agent in. It opens the ailoy docs as an example.")
    await agent.startBrowser()
    console.log('The agent has a window of its own in your browser.\n')

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
    await console_?.close()
  }
}

loadDotenv()
await main(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5', !process.argv.slice(2).includes('--noshell'))
