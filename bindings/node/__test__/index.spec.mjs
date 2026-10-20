import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

const require = createRequire(import.meta.url)
const ailoy = require('../index.js')
const virtx = require('@brekkylab/virtx')
const {
  Agent,
  AgentBuilder,
  LangModel,
  registerTool,
  removeLangModelProvider,
  setLangModelProvider,
} = ailoy
const { ConsoleClient, Recipe, ensureVirtx } = virtx

// ---- the console, from @brekkylab/virtx -----------------------------------------------------

test('virtx is a dependency, not re-exported', () => {
  assert.equal(ailoy.ConsoleClient, undefined)
  assert.equal(typeof ConsoleClient, 'function')
})

test('an agent takes only a virtx console', () => {
  assert.throws(() => new AgentBuilder('any/model').console({}), { code: 'INVALID_ARG' })
})

test('a console without a server fails with virtx’s code', async () => {
  // The builder looks for the server under $VIRTX_HOME/bin when it is made, so an empty home
  // is a host with no server, whatever this one has.
  const home = process.env.VIRTX_HOME
  process.env.VIRTX_HOME = mkdtempSync(join(tmpdir(), 'ailoy-no-server-'))
  try {
    await assert.rejects(ConsoleClient.builder().build(), { code: 'VIRTX_ERROR' })
  } finally {
    if (home === undefined) delete process.env.VIRTX_HOME
    else process.env.VIRTX_HOME = home
  }
})

test('a console the server refuses fails with the refusal’s code', async () => {
  await ensureVirtx()
  // A session boots on an image, and this builder names none.
  await assert.rejects(ConsoleClient.builder().build(), { code: 'INVALID_PARAMS', message: /image/ })
})

// ---- the builder and the registries --------------------------------------------------------

test('a builder is spent by build()', async () => {
  const builder = new AgentBuilder('no-such-org/model')
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
 * An OpenAI-compatible chat-completions server that asks for `call` (`add(2, 3)` unless told
 * otherwise) and then says what the tool answered — in one JSON body, or as a stream of events
 * when asked for one, as vLLM always does.
 */
const fakeModel = (call = { name: 'add', arguments: { a: 2, b: 3 } }) =>
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
    server.listen(0, '127.0.0.1', () => resolve(server))
  })

/**
 * Run the models of `org` on the vLLM backend, against `server`. vLLM reads its server from
 * the environment when a model is built, so the agent or model is built right after.
 */
const serve = (org, server) => {
  process.env.VLLM_BASE_URL = `http://127.0.0.1:${server.address().port}/v1`
  setLangModelProvider(org, 'vllm')
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
  serve('fake-throws', server)
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
  serve('fake-break', server)
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

const PROGRESS = {
  name: 'progress',
  description: 'Report progress on a city.',
  parameters: { type: 'object', properties: { city: { type: 'string' } } },
}

const toolDeltas = (deltas) => deltas.filter((d) => d.delta.role === 'tool')

test('an async generator tool streams its result', async (t) => {
  const server = await fakeModel({ name: 'progress', arguments: { city: 'Seoul' } })
  t.after(() => server.close())
  serve('fake-progress', server)
  let goOn
  const gate = new Promise((resolve) => (goOn = resolve))
  const desc = registerTool(PROGRESS, async function* ({ city }) {
    yield `looking up ${city}, `
    await gate
    yield 'done'
    yield { city }
  })

  const agent = await new AgentBuilder('fake-progress/model').tool(desc).build()
  const deltas = []
  for await (const delta of agent.runStream('progress?')) {
    deltas.push(delta)
    // The first piece arrives while the generator still waits to yield the next.
    if (delta.delta.role === 'tool') goOn()
  }

  const pieces = toolDeltas(deltas)
  assert.deepEqual(pieces[0].delta.contents, [{ type: 'text', text: 'looking up Seoul, ' }])
  assert.equal(pieces[0].finish_reason, null)
  assert.ok(pieces.every((p) => p.delta.id === 'call_1'))
  assert.equal(pieces[pieces.length - 1].finish_reason.type, 'stop')

  const result = agent.history.find((m) => m.role === 'tool')
  assert.deepEqual(result.contents, [
    { type: 'text', text: 'looking up Seoul, done' },
    { type: 'value', value: { city: 'Seoul' } },
  ])
})

test('letting go of a turn returns a generator tool', { timeout: 5000 }, async (t) => {
  const server = await fakeModel({ name: 'ticker', arguments: {} })
  t.after(() => server.close())
  serve('fake-ticker', server)
  let returned
  const closed = new Promise((resolve) => (returned = resolve))
  const desc = registerTool({ ...PROGRESS, name: 'ticker' }, async function* () {
    try {
      for (let i = 0; ; i++) {
        yield `tick ${i} `
        await new Promise((resolve) => setTimeout(resolve, 10))
      }
    } finally {
      returned()
    }
  })

  const agent = await new AgentBuilder('fake-ticker/model').tool(desc).build()
  for await (const delta of agent.runStream('tick')) {
    if (delta.delta.role === 'tool') break
  }
  await closed
})

test('a generator tool that fails to clean up does not take the process down', { timeout: 5000 }, async (t) => {
  const server = await fakeModel({ name: 'sloppy', arguments: {} })
  t.after(() => server.close())
  serve('fake-sloppy', server)
  let returned
  const closed = new Promise((resolve) => (returned = resolve))
  const desc = registerTool({ ...PROGRESS, name: 'sloppy' }, async function* () {
    try {
      for (;;) {
        yield 'tick '
        await new Promise((resolve) => setTimeout(resolve, 10))
      }
    } finally {
      returned()
      // eslint-disable-next-line no-unsafe-finally
      throw new Error('cleanup failed')
    }
  })

  const agent = await new AgentBuilder('fake-sloppy/model').tool(desc).build()
  for await (const delta of agent.runStream('tick')) {
    if (delta.delta.role === 'tool') break
  }
  await closed
  // An unhandled rejection would end the process before this.
  await new Promise((resolve) => setTimeout(resolve, 50))
})

test('a generator tool that yields what a message cannot hold is still returned', { timeout: 5000 }, async (t) => {
  const server = await fakeModel({ name: 'symbol', arguments: {} })
  t.after(() => server.close())
  serve('fake-symbol', server)
  let returned
  const closed = new Promise((resolve) => (returned = resolve))
  const desc = registerTool({ ...PROGRESS, name: 'symbol' }, async function* () {
    try {
      yield 'counting '
      yield Symbol('not a value')
      yield 'never'
    } finally {
      returned()
    }
  })

  const agent = await new AgentBuilder('fake-symbol/model').tool(desc).build()
  for await (const _ of agent.runStream('count')) {
  }
  await closed
  const result = agent.history.find((m) => m.role === 'tool')
  assert.equal(result.contents[0].text, 'counting ')
  assert.match(result.contents[1].value, /^error: /)
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

test('a custom provider serves an agent', async (t) => {
  const seen = []
  setLangModelProvider('mine', (org, name) => {
    seen.push([org, name])
    return new Echo(name)
  })
  t.after(() => removeLangModelProvider('mine'))

  const agent = await new AgentBuilder('mine/echo').build()
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

test('a factory that throws fails the build', async (t) => {
  setLangModelProvider('broken', () => {
    throw new Error('no models today')
  })
  t.after(() => removeLangModelProvider('broken'))
  await assert.rejects(new AgentBuilder('broken/x').build(), { message: /no models today/ })
})

// ---- a shared console, against the console server virtx starts by default ---------------

test('an agent shares its console, and closing the agent leaves it open', async () => {
  const console_ = await ConsoleClient.builder()
    .image(new Recipe('python:3.12-slim-trixie'))
    .network(false)
    .build()
  try {
    // Never asked anything, so the model only has to be built.
    process.env.VLLM_BASE_URL = 'http://127.0.0.1:9/v1'
    setLangModelProvider('fake-console', 'vllm')
    const agent = await new AgentBuilder('fake-console/model').shellTool().console(console_).build()
    await agent.close()
    const result = await console_.exec(['echo', 'still here'])
    assert.equal(result.stdout.toString().trim(), 'still here')
  } finally {
    await console_.close()
  }
})

test('an agent runs its console tools in the shared session, until the console closes', async (t) => {
  const server = await fakeModel({ name: 'shell', arguments: { cmd: 'echo relayed > /tmp/out.txt' } })
  t.after(() => server.close())

  const console_ = await ConsoleClient.builder()
    .image(new Recipe('python:3.12-slim-trixie'))
    .network(false)
    .build()
  serve('fake-shell', server)
  const agent = await new AgentBuilder('fake-shell/model').shellTool().console(console_).build()
  const later = await Agent.fromSpec({ model: 'fake-shell/model', tools: [] }, { console: console_ })
  try {
    const outputs = []
    for await (const output of agent.run('write it')) outputs.push(output)
    assert.equal(outputs[1].message.role, 'tool')
    const result = await console_.exec(['cat', '/tmp/out.txt'])
    assert.equal(result.stdout.toString(), 'relayed\n')
  } finally {
    await console_.close()
  }

  // The session was the console's: an agent attached to it fails once it is closed.
  const shell = await new AgentBuilder('fake-shell/model').shellTool().console(console_).build()
  await assert.rejects(async () => {
    for await (const _ of shell.run('write it again'));
  }, { code: 'CONSOLE_BROKEN', message: /closed/ })
  await agent.close()
  await later.close()
})
