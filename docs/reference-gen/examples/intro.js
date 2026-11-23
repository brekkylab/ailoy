const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5')
    .instruction('Answer in one short paragraph.')
    .webFetchTool()
    .build()

  for await (const out of agent.run('What is on https://example.com?')) {
    for (const part of out.message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
  await agent.close()
}

main()
