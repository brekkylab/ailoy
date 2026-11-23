const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').webFetchTool().build()

  const turn = agent.run('Read https://example.com and summarize it.') // nothing has run yet
  for await (const out of turn) {
    console.log(out.message.role, out.finish_reason.type)
    if (out.message.role === 'tool') break // ends the turn here, after the first tool result
  }
  await agent.close()
}

main()
