const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').build()

  // run(): each whole message, once it's complete
  for await (const out of agent.run('Name three prime numbers.')) {
    for (const part of out.message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }

  // runStream(): the model's words as they're written. This turn sees the last one.
  for await (const out of agent.runStream('Now add them up.')) {
    for (const piece of out.delta.contents) process.stdout.write(piece.text ?? '')
  }
  console.log()

  console.log(agent.history.length, 'messages so far')
  await agent.close()
}

main()
