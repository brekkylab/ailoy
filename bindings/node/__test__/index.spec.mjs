import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { test } from 'node:test'

const ailoy = createRequire(import.meta.url)('../index.js')
const { Agent, AgentBuilder, Console, Image, registerLangModel, registerTool } = ailoy

// ---- cortex, built in ----------------------------------------------------------------------

test('cortex comes along', () => {
  assert.match(new Image('python:3.12-slim').step('pip install duckdb').toString(), /duckdb/)
})

test('a console without a server fails with cortex’s code', async () => {
  await assert.rejects(Console.builder().build(), { code: 'CORTEX_ERROR' })
})

// ---- the builder and the registries --------------------------------------------------------

test('a builder is spent by build()', async () => {
  const builder = new AgentBuilder('no-such-provider/model')
  await assert.rejects(builder.build(), { code: 'AILOY_ERROR' })
  assert.throws(() => builder.instruction('again'), { code: 'INVALID_ARG' })
})

test('an object that does not fit says what was expected', () => {
  assert.throws(() => new AgentBuilder('m').tool({ description: 'no name' }), {
    code: 'INVALID_ARG',
    message: /not a tool description/,
  })
})

test('registering into a provider that does not exist throws', () => {
  assert.throws(() => registerTool({ name: 't', parameters: {} }, () => 1, { provider: 'nope' }), {
    code: 'AILOY_ERROR',
  })
})

// ---- a turn, against a model served from this process --------------------------------------

/** The last message's text, or the tool results it carries. */
const lastContent = (messages) => {
  const last = messages[messages.length - 1]
  return typeof last.content === 'string' ? last.content : JSON.stringify(last.content)
}

/**
 * An OpenAI-compatible chat-completions server that asks for `add(2, 3)` and then says what
 * the tool answered — in one JSON body, or as a stream of events when asked for one.
 */
const fakeModel = () =>
  new Promise((resolve) => {
    const server = createServer((req, res) => {
      let body = ''
      req.on('data', (chunk) => (body += chunk))
      req.on('end', () => {
        const { messages, stream } = JSON.parse(body)
        const answered = messages[messages.length - 1].role === 'tool'
        const message = answered
          ? { role: 'assistant', content: `the tool said ${lastContent(messages)}` }
          : {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: { name: 'add', arguments: JSON.stringify({ a: 2, b: 3 }) },
                },
              ],
            }
        const finish_reason = answered ? 'stop' : 'tool_calls'
        const usage = { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
        if (!stream) {
          res.setHeader('content-type', 'application/json')
          res.end(JSON.stringify({ choices: [{ index: 0, message, finish_reason }], usage }))
          return
        }
        res.setHeader('content-type', 'text/event-stream')
        const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`)
        const delta = { ...message }
        if (delta.tool_calls) delta.tool_calls = delta.tool_calls.map((c, index) => ({ index, ...c }))
        send({ choices: [{ index: 0, delta, finish_reason: null }] })
        send({ choices: [{ index: 0, delta: {}, finish_reason }] })
        send({ choices: [], usage })
        res.end('data: [DONE]\n\n')
      })
    })
    server.listen(0, '127.0.0.1', () => resolve(server))
  })

const ADD = {
  name: 'add',
  description: 'Add two numbers.',
  parameters: {
    type: 'object',
    properties: { a: { type: 'number' }, b: { type: 'number' } },
    required: ['a', 'b'],
  },
}

test('a turn calls a JavaScript tool and answers with what it said', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  registerLangModel('fake/*', 'chat_completion', `http://127.0.0.1:${server.address().port}/v1/chat/completions`)

  const calls = []
  const desc = registerTool(ADD, async ({ a, b }) => {
    calls.push([a, b])
    return a + b
  })

  const agent = await new AgentBuilder('fake/model').tool(desc).build()
  try {
    const outputs = []
    for await (const output of agent.run('what is 2 + 3?')) outputs.push(output)

    assert.deepEqual(calls, [[2, 3]])
    assert.deepEqual(
      outputs.map((o) => o.message.role),
      ['assistant', 'tool', 'assistant'],
    )
    assert.equal(outputs[1].message.contents[0].value, 5)
    assert.match(outputs[2].message.contents[0].text, /5/)
    assert.equal(agent.history.length, 4)

    // The same turn, streamed.
    const deltas = []
    for await (const delta of agent.runStream('and again?')) deltas.push(delta)
    assert.ok(deltas.some((d) => d.finish_reason?.type === 'tool_call'))
    assert.deepEqual(calls, [[2, 3], [2, 3]])
  } finally {
    await agent.close()
  }
  await assert.rejects(agent.run('closed').next(), { code: 'AILOY_ERROR' })
})

test('a tool that throws answers with the error, and the turn goes on', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  registerLangModel('fake-throws/*', 'chat_completion', `http://127.0.0.1:${server.address().port}/v1/chat/completions`)
  const desc = registerTool(ADD, () => {
    throw new Error('no adding today')
  })

  const agent = await Agent.fromSpec({ model: 'fake-throws/model', tools: [desc] })
  const outputs = []
  for await (const output of agent.run({ role: 'user', contents: [{ type: 'text', text: '2 + 3?' }] })) {
    outputs.push(output)
  }
  assert.equal(outputs[1].message.contents[0].value, 'error: no adding today')
  assert.equal(outputs.length, 3)
})

test('breaking out of a turn ends it where it stands', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  registerLangModel('fake-break/*', 'chat_completion', `http://127.0.0.1:${server.address().port}/v1/chat/completions`)
  const calls = []
  const desc = registerTool(ADD, ({ a, b }) => {
    calls.push([a, b])
    return a + b
  })

  const agent = await new AgentBuilder('fake-break/model').tool(desc).build()
  for await (const output of agent.run('2 + 3?')) {
    assert.equal(output.message.role, 'assistant')
    break
  }
  assert.deepEqual(calls, [])
  // The agent is free again for the next turn.
  assert.ok(Array.isArray(agent.history))
})

// ---- a shared console, against a real console server named by `$CORTEX_CONSOLE` ------------

const SERVER = process.env.CORTEX_CONSOLE

test('an agent shares its console, and closing the agent leaves it open', { skip: !SERVER && 'set $CORTEX_CONSOLE' }, async () => {
  const console_ = await Console.builder()
    .stdioClient([SERVER])
    .image(new Image('python:3.12-slim-trixie'))
    .network(ailoy.NetworkAccess.none())
    .build()
  try {
    // Never asked anything, so the model only has to be registered.
    registerLangModel('fake-console/*', 'chat_completion', 'http://127.0.0.1:9/v1/chat/completions')
    const agent = await new AgentBuilder('fake-console/model').shellTool().console(console_).build()
    await agent.close()
    const result = await console_.exec(['echo', 'still here'])
    assert.equal(result.stdout.toString().trim(), 'still here')
  } finally {
    await console_.close()
  }
})
