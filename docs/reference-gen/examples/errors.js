const { AgentBuilder } = require('@brekkylab/ailoy')

async function main() {
  try {
    // no org called "nobody" has a model provider
    const agent = await new AgentBuilder('nobody/no-model').build()
    await agent.close()
  } catch (/** @type {any} */ e) {
    if (e.code === 'AILOY_ERROR') console.error('ailoy refused:', e.message)
    else throw e
  }
}

main()
