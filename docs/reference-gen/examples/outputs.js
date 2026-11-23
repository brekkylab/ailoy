const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').build()

  // Whole messages: why each ended, and what it cost
  for await (const out of agent.run('Explain DNS in two sentences.')) {
    if (out.usage) console.log('tokens in/out:', out.usage.input_tokens, out.usage.output_tokens)
    if (out.finish_reason.type === 'refusal') console.log('refused:', out.finish_reason.reason)
  }

  // Deltas: put the text back together until a finish_reason ends the message
  let text = ''
  for await (const out of agent.runStream('Now explain it to a child.')) {
    for (const piece of out.delta.contents) text += piece.text ?? ''
    if (out.finish_reason) {
      console.log(text, `(${out.finish_reason.type})`)
      text = ''
    }
  }
  await agent.close()
}

main()
