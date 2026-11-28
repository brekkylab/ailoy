import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { join } from 'node:path'
import { test } from 'node:test'

const require = createRequire(import.meta.url)
const ailoy = require('../index.js')
const virtx = require('@brekkylab/virtx')
const {
  Agent,
  LangModel,
  registerTool,
  getLangModelProvider,
  removeLangModelProvider,
  removeToolProvider,
  setLangModelProvider,
} = ailoy
const { Client } = virtx

// ---- the sandbox, from @brekkylab/virtx -----------------------------------------------------

test('virtx is a dependency, not re-exported', () => {
  assert.equal(ailoy.Sandbox, undefined)
  assert.equal(typeof Client, 'function')
})

test('an agent takes only a virtx sandbox', () => {
  assert.throws(() => Agent.fromSpec({ model: 'any/model', use_sandbox: true }, { sandbox: {} }), {
    code: 'INVALID_ARG',
  })
})

// ---- the registries --------------------------------------------------------------------------

test('a model nothing provides is an error', async () => {
  await assert.rejects(Agent.fromSpec({ model: 'no-such-org/model' }), { code: 'AILOY_ERROR' })
})

test('the web search provider is set by name', () => {
  assert.equal(ailoy.getWebSearchProvider(), 'web')
  try {
    ailoy.setWebSearchProvider('naver')
    assert.equal(ailoy.getWebSearchProvider(), 'naver')
    assert.throws(() => ailoy.setWebSearchProvider('bing'), { code: 'INVALID_ARG' })
  } finally {
    ailoy.setWebSearchProvider('web')
  }
})

test('an object that does not fit says what was expected', () => {
  assert.throws(() => registerTool({ description: 'no name' }, () => null), {
    code: 'INVALID_ARG',
    message: /not a tool description/,
  })
})

/**
 * Run the models of `org` on the self-hosted backend, against a port nothing listens
 * on, for an agent that is built but never asked.
 */
const serveNowhere = (org) => {
  process.env.SELF_HOSTED_BASE_URL = 'http://127.0.0.1:9/v1'
  setLangModelProvider(org, 'self-hosted')
}

test('a browser needs a sandbox with its abin', async () => {
  serveNowhere('fake-browser')
  await assert.rejects(Agent.fromSpec({ model: 'fake-browser/model', use_browser: true }), {
    code: 'AILOY_ERROR',
    message: /`browser` abin/,
  })
})

// ---- a turn, against a model served from this process --------------------------------------

/** The last message's text, or the tool results it carries. */
const lastContent = (messages) => {
  const last = messages[messages.length - 1]
  return typeof last.content === 'string' ? last.content : JSON.stringify(last.content)
}

/**
 * An OpenAI-compatible chat-completions server that asks for `call` (`add(2, 3)` unless told
 * otherwise) and then says what the tool answered — in one JSON body, or as a stream of events
 * when asked for one, as vLLM always does.
 */
const fakeModel = (call = { name: 'add', arguments: { a: 2, b: 3 } }) =>
  new Promise((resolve) => {
    const server = createServer((req, res) => {
      if (req.method === 'GET') {
        res.setHeader('content-type', 'application/json')
        res.end(JSON.stringify({ data: server.listed.map((id) => ({ id })) }))
        return
      }
      let body = ''
      req.on('data', (chunk) => (body += chunk))
      req.on('end', () => {
        const { model, messages, stream } = JSON.parse(body)
        server.seen.push([model, req.headers.authorization ?? null])
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
                  function: { name: call.name, arguments: JSON.stringify(call.arguments) },
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
    // Every request's model id and `authorization` header, as sent.
    server.seen = []
    // The ids `GET /v1/models` lists.
    server.listed = []
    server.listen(0, '127.0.0.1', () => resolve(server))
  })

/**
 * Run the models of `org` on the self-hosted backend, against `server`. The backend
 * reads its server from the environment when a model is built, so the agent or model is built
 * right after.
 */
const serve = (org, server) => {
  process.env.SELF_HOSTED_BASE_URL = `http://127.0.0.1:${server.address().port}/v1`
  setLangModelProvider(org, 'self-hosted')
}

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
  serve('fake', server)

  const calls = []
  const name = registerTool(ADD, async ({ a, b }) => {
    calls.push([a, b])
    return a + b
  })

  const agent = await Agent.fromSpec({ model: 'fake/model', tools: [name] })
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
  serve('fake-throws', server)
  const name = registerTool(ADD, () => {
    throw new Error('no adding today')
  })

  const agent = await Agent.fromSpec({ model: 'fake-throws/model', tools: [name] })
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
  serve('fake-break', server)
  const calls = []
  const name = registerTool(ADD, ({ a, b }) => {
    calls.push([a, b])
    return a + b
  })

  const agent = await Agent.fromSpec({ model: 'fake-break/model', tools: [name] })
  for await (const output of agent.run('2 + 3?')) {
    assert.equal(output.message.role, 'assistant')
    break
  }
  assert.deepEqual(calls, [])
  // The agent is free again for the next turn.
  assert.ok(Array.isArray(agent.history))
})

test('a server of your own keeps the desc', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  const base_url = `http://127.0.0.1:${server.address().port}/v1`
  setLangModelProvider('meta-llama', {
    backend: 'self-hosted',
    base_url,
    api_key: 'k',
    models: { 'Llama-3.3-70B-Instruct': 'llama3.3:70b' },
  })
  setLangModelProvider('Qwen', { backend: 'self-hosted', base_url })
  try {
    assert.equal(getLangModelProvider('meta-llama/Llama-3.3-70B-Instruct'), 'self-hosted')
    server.listed = ['qwen/qwen3-8b', 'qwen3.5-27b']
    const messages = [{ role: 'user', contents: [{ type: 'text', text: 'hi' }] }]
    for (const desc of [
      'meta-llama/Llama-3.3-70B-Instruct',
      'meta-llama/Llama-3.1-8B-Instruct',
      'Qwen/Qwen3-8B',
      'Qwen/Qwen3.5-27B',
    ]) {
      await (await LangModel.create(desc)).infer(messages)
    }
    // The server is sent its own id for the model: mapped by `models`, listed by the server
    // for the desc or the name alone, or the desc as it is.
    assert.deepEqual(server.seen, [
      ['llama3.3:70b', 'Bearer k'],
      ['meta-llama/Llama-3.1-8B-Instruct', 'Bearer k'],
      ['qwen/qwen3-8b', null],
      ['qwen3.5-27b', null],
    ])
  } finally {
    removeLangModelProvider('meta-llama')
    setLangModelProvider('Qwen', 'api')
  }

  assert.throws(() => setLangModelProvider('x', { backend: 'cli', base_url }), {
    code: 'INVALID_ARG',
    message: /self-hosted/,
  })
  assert.throws(() => setLangModelProvider('x', { backend: 'self-hosted' }), { code: 'INVALID_ARG', message: /base_url/ })
  assert.throws(() => setLangModelProvider('x', { backend: 'self-hosted', baseUrl: base_url }), {
    code: 'INVALID_ARG',
    message: /baseUrl/,
  })
  assert.throws(() => setLangModelProvider('x', { backend: 'self-hosted', base_url: '' }), {
    code: 'INVALID_ARG',
    message: /base_url/,
  })
  assert.throws(() => setLangModelProvider('x', { backend: 'self-hosted', base_url, models: ['a'] }), {
    code: 'INVALID_ARG',
  })
  assert.equal(getLangModelProvider('x'), null)
})

test('the self-hosted backend reads its server from the environment', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  setLangModelProvider('env-served', 'self-hosted')
  const messages = [{ role: 'user', contents: [{ type: 'text', text: 'hi' }] }]
  try {
    delete process.env.SELF_HOSTED_BASE_URL
    await assert.rejects(LangModel.create('env-served/model'), { message: /SELF_HOSTED_BASE_URL/ })

    process.env.SELF_HOSTED_BASE_URL = `http://127.0.0.1:${server.address().port}/v1`
    await (await LangModel.create('env-served/model')).infer(messages)
    assert.deepEqual(server.seen, [['env-served/model', null]])
  } finally {
    delete process.env.SELF_HOSTED_BASE_URL
    removeLangModelProvider('env-served')
  }
})

test('a model runs without an agent', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  serve('fake-model', server)
  const model = await LangModel.create('fake-model/model')

  const messages = [{ role: 'user', contents: [{ type: 'text', text: '2 + 3?' }] }]
  const output = await model.infer(messages, [ADD])
  assert.equal(output.finish_reason.type, 'tool_call')
  assert.equal(output.message.tool_calls[0].function.name, 'add')

  const deltas = []
  for await (const delta of model.inferStream(messages, [ADD])) deltas.push(delta)
  assert.ok(deltas.some((d) => d.finish_reason?.type === 'tool_call'))
})

test('a backend that does not exist is refused', () => {
  assert.throws(() => setLangModelProvider('x', 'smoke-signals'), { code: 'INVALID_ARG' })
  assert.throws(() => setLangModelProvider('x', 42), { code: 'INVALID_ARG' })
})

/** A model of the test's own, answering with the last user text. */
class Echo {
  constructor(name) {
    this.name = name
  }
  infer(messages) {
    const text = messages[messages.length - 1].contents[0].text
    return {
      message: { role: 'assistant', contents: [{ type: 'text', text: `${this.name}: ${text}` }] },
      finish_reason: { type: 'stop' },
    }
  }
}

class StreamingEcho extends Echo {
  async infer(messages) {
    await null
    return super.infer(messages)
  }
  async *inferStream(messages) {
    for (const text of [this.name, ': ', messages[messages.length - 1].contents[0].text]) {
      yield { delta: { role: 'assistant', contents: [{ type: 'text', text }], tool_calls: [] } }
    }
    yield { delta: { contents: [], tool_calls: [] }, finish_reason: { type: 'stop' } }
  }
}

const textOf = (outputs) => outputs.map((o) => o.message.contents[0].text)

test('a provider reads back as it was set', () => {
  assert.equal(getLangModelProvider('anthropic'), 'cli')
  assert.equal(getLangModelProvider('anthropic/claude-sonnet-5-5'), 'cli')
  setLangModelProvider('read-back', 'bedrock')
  assert.equal(getLangModelProvider('read-back/some-model'), 'bedrock')
  setLangModelProvider('read-back', (org, name) => new Echo(name))
  assert.equal(getLangModelProvider('read-back'), 'custom')
  removeLangModelProvider('read-back')
  assert.equal(getLangModelProvider('read-back'), null)
})

test('a custom provider serves an agent', async (t) => {
  const seen = []
  setLangModelProvider('mine', (org, name) => {
    seen.push([org, name])
    return new Echo(name)
  })
  t.after(() => removeLangModelProvider('mine'))

  const agent = await Agent.fromSpec({ model: 'mine/echo' })
  const outputs = []
  for await (const output of agent.run('hi')) outputs.push(output)
  assert.deepEqual(textOf(outputs), ['echo: hi'])

  const deltas = []
  for await (const delta of agent.runStream('again')) deltas.push(delta)
  assert.equal(deltas[deltas.length - 1].finish_reason.type, 'stop')
  assert.equal(agent.history[agent.history.length - 1].contents[0].text, 'echo: again')
  assert.deepEqual(seen, [['mine', 'echo']])
})

test('a custom provider may stream and be async', async (t) => {
  setLangModelProvider('mine-async', (org, name) => new StreamingEcho(name ?? org))
  t.after(() => removeLangModelProvider('mine-async'))

  const model = await LangModel.create('mine-async')
  const messages = [{ role: 'user', contents: [{ type: 'text', text: 'hi' }] }]
  assert.equal((await model.infer(messages)).message.contents[0].text, 'mine-async: hi')
  const deltas = []
  for await (const delta of model.inferStream(messages)) deltas.push(delta)
  assert.equal(deltas.length, 4)
})

test('a custom provider may hand back a built-in model', async (t) => {
  const server = await fakeModel()
  t.after(() => server.close())
  serve('fake-alias', server)
  const built = await LangModel.create('fake-alias/model')
  setLangModelProvider('alias', () => built)
  t.after(() => removeLangModelProvider('alias'))

  const model = await LangModel.create('alias/any')
  const messages = [{ role: 'user', contents: [{ type: 'text', text: '2 + 3?' }] }]
  assert.equal((await model.infer(messages, [ADD])).finish_reason.type, 'tool_call')
})

test('a removed tool provider no longer builds', async (t) => {
  setLangModelProvider('tooled', (org, name) => new Echo(name))
  t.after(() => removeLangModelProvider('tooled'))
  const name = registerTool({ name: 'fleeting', parameters: { type: 'object' } }, () => null)

  await Agent.fromSpec({ model: 'tooled/x', tools: [name] })
  assert.equal(removeToolProvider(name), true)
  assert.equal(removeToolProvider(name), false)
  await assert.rejects(Agent.fromSpec({ model: 'tooled/x', tools: [name] }), { message: /fleeting/ })
})

test('a factory that throws fails the build', async (t) => {
  setLangModelProvider('broken', () => {
    throw new Error('no models today')
  })
  t.after(() => removeLangModelProvider('broken'))
  await assert.rejects(Agent.fromSpec({ model: 'broken/x' }), { message: /no models today/ })
})

// ---- the caller's sandbox, against the server virtx starts by default ---------------------

test('an agent runs its console tools in the caller’s sandbox, until it is removed', async (t) => {
  const server = await fakeModel({ name: 'shell', arguments: { cmd: 'echo shared > /tmp/out.txt' } })
  t.after(() => server.close())
  const client = Client.spawn()
  t.after(() => client.close())
  await client.pullImage('python:3.12-slim').complete()
  const sandbox = await client.createSandbox('python:3.12-slim', { network: false, memoryMib: 1024 })

  serve('fake-shell', server)
  const agent = await Agent.fromSpec({ model: 'fake-shell/model', use_sandbox: true }, { sandbox })
  const outputs = []
  for await (const output of agent.run('write it')) outputs.push(output)
  assert.equal(outputs[1].message.role, 'tool')
  // The sandbox is still the caller's to use.
  assert.equal(await sandbox.readText('tmp/out.txt'), 'shared\n')

  // Removed, it is gone for an agent too.
  const later = await Agent.fromSpec({ model: 'fake-shell/model', use_sandbox: true }, { sandbox })
  const fromSpec = await Agent.fromSpec({ model: 'fake-shell/model', use_sandbox: true }, { sandbox })
  await sandbox.remove()
  const after = []
  for await (const output of later.run('write it again')) after.push(output)
  assert.match(JSON.stringify(after[1].message.contents), /404/)
  await agent.close()
  await later.close()
  await fromSpec.close()
})
