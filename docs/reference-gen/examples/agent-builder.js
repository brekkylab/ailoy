const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5')
    .instruction('You are a research assistant. Name the pages you read.')
    .webSearchTool(['DuckDuckGo', 'Bing'])
    .webFetchTool()
    .thinkingEffort('medium')
    .contextManager({ maxInputTokens: 100_000 })
    .build()

  for await (const out of agent.run('What changed in the latest Python release?')) {
    for (const part of out.message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
  await agent.close()
}

main()
