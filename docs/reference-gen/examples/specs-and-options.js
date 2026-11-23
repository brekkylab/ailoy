const { Agent, AgentBuilder } = require('@brekkylab/ailoy')

// An agent as data: store it as JSON, load it, build it
const spec = {
  model: 'anthropic/claude-sonnet-5-5',
  instruction: 'Be brief.',
  model_options: { thinking_effort: 'low' },
}
const saved = JSON.stringify(spec)

async function main() {
  const agent = await Agent.fromSpec(JSON.parse(saved))
  await agent.close()

  // The options objects the builder takes
  const browsing = await new AgentBuilder('anthropic/claude-sonnet-5-5')
    .browser({ isolated: true, headless: true })
    .contextManager({ maxInputTokens: 100_000, preserveRecentTurns: 2 })
    .build()
  await browsing.close()
}

main()
