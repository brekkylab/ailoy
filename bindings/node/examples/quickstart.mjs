// An agent with a tool of its own, run for one turn.
//
// The model is resolved from the default provider, which is seeded from the API keys in the
// environment — `ANTHROPIC_API_KEY` for the model named here.
//
//     ANTHROPIC_API_KEY=... node examples/quickstart.mjs

import { createRequire } from 'node:module'

const { AgentBuilder, registerTool } = createRequire(import.meta.url)('../index.js')

// A tool is its description and a function. The model's arguments arrive as one object; what
// the function returns, or resolves to, is the tool's result.
const weather = registerTool(
  {
    name: 'weather',
    description: 'The current weather in a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
  },
  async ({ city }) => ({ city, sky: 'clear', celsius: 21 }),
)

const agent = await new AgentBuilder('anthropic/claude-sonnet-5')
  .instruction('Answer in one sentence.')
  .tool(weather)
  .build()

try {
  // Deltas as the model writes them; each tool's result arrives whole.
  for await (const { delta } of agent.runStream('What is the weather in Seoul?')) {
    for (const part of delta.contents) {
      if (part.type === 'text') process.stdout.write(part.text)
    }
  }
  process.stdout.write('\n')
} finally {
  await agent.close()
}
