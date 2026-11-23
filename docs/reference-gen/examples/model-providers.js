const { LangModel, getLangModelProvider, setLangModelProvider } = require('@brekkylab/ailoy')

// Claude through its own API (reads ANTHROPIC_API_KEY) instead of Claude Code
setLangModelProvider('anthropic', 'api')
console.log(getLangModelProvider('anthropic/claude-sonnet-5-5')) // api

// Qwen on your own vLLM server (VLLM_BASE_URL)
setLangModelProvider('Qwen', 'vllm')

// A model of your own: here, one that repeats the last message back
/** @type {import('@brekkylab/ailoy').CustomLangModel} */
const echo = {
  infer(messages) {
    const last = messages[messages.length - 1].contents
    return { message: { role: 'assistant', contents: last }, finish_reason: { type: 'stop' } }
  },
}
setLangModelProvider('echo', () => echo)

async function main() {
  const model = await LangModel.create('echo/any-name')
  const out = await model.infer([{ role: 'user', contents: [{ type: 'text', text: 'hi' }] }])
  console.log(out.message.contents)
}

main()
