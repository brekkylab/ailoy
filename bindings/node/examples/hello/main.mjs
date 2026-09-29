// Run one agent turn with a fixed question: no tools, no console, no system message.
//
//     node examples/hello/main.mjs [model]
//
// `model` defaults to `bedrock/global.openai.gpt-5.6-luna`; its provider's API key has to be
// set (`AWS_BEARER_TOKEN_BEDROCK`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), in the
// environment or in `.env`.
//
// The addon has to be built first (`npm run build` in `bindings/node`).

import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))

const { AgentBuilder } = createRequire(import.meta.url)('../../index.js')

// The request the agent is given.
const QUERY = 'What is the meaning of hello world?'

// From the nearest `.env` up from this file, as the Rust and Python examples load it. What the
// environment already has wins.
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
await main(process.argv[2] ?? 'bedrock/global.openai.gpt-5.6-luna')
