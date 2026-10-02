// Run one agent turn with a fixed question: no tools, no console, no system message.
//
//     node main.mjs [model]
//
// `model` defaults to `openai/gpt-5.6-luna`; its provider's API key has to be
// set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), in the
// environment or in `.env`.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import ailoy from 'ailoy-node'

const HERE = dirname(fileURLToPath(import.meta.url))

const { AgentBuilder } = ailoy

const QUERY = 'What is the meaning of hello world?'

// From the nearest `.env` up from this file. What the environment already has wins.
function loadDotenv() {
  for (let dir = HERE; ; dir = dirname(dir)) {
    const path = join(dir, '.env')
    if (existsSync(path)) return process.loadEnvFile(path)
    if (dirname(dir) === dir) return
  }
}

async function main(model) {
  const agent = await new AgentBuilder(model).build()
  console.log(`model  ${model}\n`)

  try {
    for await (const { message } of agent.run(QUERY)) {
      if (message.role === 'assistant') {
        for (const part of message.contents) {
          if (part.type === 'text') console.log(part.text)
        }
      }
    }
  } finally {
    await agent.close()
  }
}

loadDotenv()
await main(process.argv[2] ?? 'openai/gpt-5.6-luna')
