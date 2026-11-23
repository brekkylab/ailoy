const { LangModel } = require('@brekkylab/ailoy')

async function main() {
  const model = await LangModel.create('anthropic/claude-sonnet-5-5')
  const out = await model.infer(
    [
      { role: 'system', contents: [{ type: 'text', text: 'Reply in French.' }] },
      { role: 'user', contents: [{ type: 'text', text: 'Good morning!' }] },
    ],
    [],
    { thinking_effort: 'low' },
  )
  for (const part of out.message.contents) {
    if (part.type === 'text') console.log(part.text)
  }
}

main()
