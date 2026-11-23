const { AgentBuilder, registerMcpStdio, registerTool } = require('@brekkylab/ailoy')

// Your own function: the desc is what the model sees, the function what runs
const weather = registerTool(
  {
    name: 'get_weather',
    description: 'The current weather in a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
  },
  ({ city }) => `Sunny and 21°C in ${city}`,
)

async function main() {
  // An MCP server's tools, named fs__<tool>
  const fsTools = await registerMcpStdio('fs', 'npx', ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])

  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').tool(weather).tools(fsTools).build()
  for await (const out of agent.run("What's the weather in Seoul?")) {
    console.log(out.message.role, out.message.contents)
  }
  await agent.close()
}

main()
