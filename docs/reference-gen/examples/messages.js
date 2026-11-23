const { AgentBuilder } = require('@brekkylab/ailoy')

/** @type {import('@brekkylab/ailoy').Message} */
const question = {
  role: 'user',
  contents: [
    { type: 'text', text: 'What is in this picture?' },
    { type: 'image', image: { type: 'url', url: 'https://example.com/cat.png' } },
  ],
}

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').build()
  for await (const { message } of agent.run(question)) {
    for (const call of message.tool_calls ?? []) {
      console.log('calls', call.function.name, call.function.arguments)
    }
    for (const part of message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
  await agent.close()
}

main()
