// An OpenAI-compatible chat-completions endpoint on localhost, enough to drive a whole turn
// -- model, tool, model -- without a network or a key. `python.py` serves the same one.
//
// Given tools, it calls the one the user's message names, with the arguments after the name
// (`add {"a": 2, "b": 3}`); given a tool's result, it says what the tool said; otherwise it
// says hello.
import { createServer } from 'node:http'

const text = (content) => (typeof content === 'string' ? content : (content ?? []).map((p) => p.text ?? '').join(''))

export const fakeModel = () =>
  new Promise((resolve) => {
    const server = createServer((req, res) => {
      let body = ''
      req.on('data', (chunk) => (body += chunk))
      req.on('end', () => {
        const { messages, tools } = JSON.parse(body)
        const last = messages[messages.length - 1]
        let message
        if (last.role === 'tool') {
          message = { role: 'assistant', content: `the tool said ${text(last.content)}` }
        } else if (tools?.length) {
          const [name, ...args] = text(messages.find((m) => m.role === 'user').content).split(' ')
          message = {
            role: 'assistant',
            content: null,
            tool_calls: [{ id: 'call_1', type: 'function', function: { name, arguments: args.join(' ') } }],
          }
        } else {
          message = { role: 'assistant', content: 'hello' }
        }
        const finish_reason = message.tool_calls ? 'tool_calls' : 'stop'
        res.setHeader('content-type', 'application/json')
        res.end(
          JSON.stringify({
            choices: [{ index: 0, message, finish_reason }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
        )
      })
    })
    server.listen(0, '127.0.0.1', () =>
      resolve({
        url: `http://127.0.0.1:${server.address().port}/v1/chat/completions`,
        close: () => new Promise((r) => server.close(r)),
      }),
    )
  })
