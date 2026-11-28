// An agent with a browser, waiting for what you ask of it.
//
//     node main.mjs
//
// The agent works in a sandbox with python and curl and virtx's `browser` abin, a headless
// Chromium, which it drives with its `browser` and `browser_screenshot` tools; the shell and file
// tools are there too, for what is easier done without the browser, such as fetching a page's
// data or computing over it. Each line you type is the next turn; `/exit` ends.
//
// `AILOY_MODEL` is the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend has
// to be set up as `docs/content/guides/model-providers.mdx` says, any key in the
// environment or in `.env`.
//
// `npm install` here first, after building the addon (`npm run build` in `ailoy-node`).

import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'
import virtx from '@brekkylab/virtx'

const HERE = dirname(fileURLToPath(import.meta.url))

const { AgentBuilder } = ailoy
const { Client, Recipe } = virtx

// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE = 'ailoy-browser-use:1'

// What the agent has.
const HAS = 'a sandbox and a web browser in it, which you drive with the browser tools'

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

async function main(model) {
  // The virtx server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none. The sandboxes it holds go when it is closed.
  const client = Client.spawn()

  // The image, built once: python, and curl on it.
  // The server answers a tag in its full form, as `docker.io/library/ailoy-browser-use:1`.
  if (!(await client.listImage()).some((image) => image.reference.split('/').pop() === IMAGE)) {
    console.log(`Building ${IMAGE}, once; it takes a few minutes.`)
    const recipe = new Recipe('python:3.12-slim-trixie').run(
      'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*',
    )
    await client.buildImage(recipe, IMAGE).complete()
  }

  // The browser is the sandbox's: its abin, a network for the pages it opens, and the memory
  // Chromium takes.
  const sandbox = await client.createSandbox(IMAGE, { network: true, abins: ['browser'], memoryMib: 2048 })
  const agent = await new AgentBuilder(model)
    .instruction(`You are a helpful assistant with ${HAS}. ${INSTRUCTION}`)
    .sandbox(sandbox)
    .browser()
    .build()
  console.log(`model     ${model}\n`)
  try {
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
    await client.close()
  }
}

loadDotenv()
await main(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5')
