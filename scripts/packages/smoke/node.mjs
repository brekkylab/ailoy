// The Node package as a user installs it, on the platform this runs on.
//
// Run from a project that installed `@brekkylab/ailoy` and `@brekkylab/virtx`, so each resolves
// the way that project's code would. What this platform can do is said in the environment:
//   SMOKE_SERVER  1 if virtx publishes `virtx-stdio` for this platform, which `ensureVirtx()`
//                 fetches into $VIRTX_HOME, else 0
//   SMOKE_VM      1 if this machine can boot one (KVM or HVF), else 0
//
// A turn's model is served from this process (`fake.mjs`): no network, and no API key.
import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

import { fakeModel } from './fake.mjs'

const require = createRequire(path.join(process.cwd(), 'noop.js'))
const want = (name) => process.env[name] === '1'
const failures = []
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`)
  if (!ok) failures.push(what)
}
const run = async (agent, query) => {
  const outputs = []
  for await (const output of agent.run(query)) outputs.push(output)
  return outputs
}

const ailoy = require('@brekkylab/ailoy')
const virtx = require('@brekkylab/virtx')
check(typeof ailoy.AgentBuilder === 'function', 'the package loads')

// Only this platform's binary, as npm's os/cpu/libc filter chose it.
const scope = path.join(process.cwd(), 'node_modules', '@brekkylab')
const installed = fs.readdirSync(scope).sort()
console.log(`  installed: ${installed.join(' ')}`)
check(installed.length === 4, 'exactly one platform package was installed beside each root (ailoy, virtx)')

check(JSON.stringify(new virtx.Recipe('alpine:latest').run('echo hi')).includes('echo hi'), 'virtx comes along')

const model = await fakeModel()
try {
  // The `fake` org runs on the self-hosted backend, which reads its server from the
  // environment.
  process.env.SELF_HOSTED_BASE_URL = model.url
  ailoy.setLangModelProvider('fake', 'self-hosted')

  // A whole turn -- model, tool, model -- through the addon's async bridge both ways.
  const calls = []
  const add = ailoy.registerTool(
    {
      name: 'add',
      description: 'Add two numbers.',
      parameters: { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } }, required: ['a', 'b'] },
    },
    async ({ a, b }) => {
      calls.push([a, b])
      return a + b
    },
  )
  const agent = await new ailoy.AgentBuilder('fake/model').tool(add).build()
  const outputs = await run(agent, 'add {"a": 2, "b": 3}')
  console.log(`  turn: ${outputs.map((o) => o.message.role).join(' -> ')}`)
  check(JSON.stringify(calls) === '[[2,3]]', 'the model called a JavaScript tool')
  check(outputs.map((o) => o.message.role).join() === 'assistant,tool,assistant', 'a turn goes model, tool, model')
  check(/5/.test(JSON.stringify(outputs.at(-1)?.message.contents)), 'the model answered with what the tool said')
  await agent.close()

  // The console server, as the guides have it: `ensureVirtx()` first.
  let server = null
  try {
    server = await virtx.ensureVirtx()
    console.log(`  ensureVirtx: ${server}`)
  } catch (e) {
    console.log(`  ensureVirtx: ${e.message}`)
  }
  check(!!server === want('SMOKE_SERVER'), `ensureVirtx ${server ? 'fetched' : 'fetched no'} server`)
  if (server) {
    const exe = process.platform === 'win32' ? '.exe' : ''
    check(path.basename(server) === `virtx-stdio${exe}` && fs.statSync(server).isFile(), `ensureVirtx answers virtx-stdio${exe}`)
    if (process.env.VIRTX_HOME) {
      const within = path.relative(process.env.VIRTX_HOME, server)
      check(!within.startsWith('..') && !path.isAbsolute(within), 'into $VIRTX_HOME')
    }
    // The server runs on this machine, and answers -- no VM needed to list its images.
    const client = virtx.Client.spawn(server)
    const images = await client.listImage()
    await client.close()
    console.log(`  listImage: ${images.length} images`)
    check(Array.isArray(images), 'the console server answers')
  }

  if (server && want('SMOKE_VM')) {
    // An agent's shell tool, in a sandbox that sees a volume.
    const client = virtx.Client.spawn(server)
    if (!(await client.listImage()).some((i) => i.reference.split('/').pop() === 'alpine:latest')) {
      await client.pullImage('alpine:latest').complete()
    }
    const volume = await client.createVolume()
    await volume.writeFile('from-host.txt', 'by volume')
    const sandbox = await client.createSandbox('alpine:latest', { volumes: [{ volume, path: '/host' }] })
    const agent = await new ailoy.AgentBuilder('fake/model').sandbox(sandbox).build()
    const cmd = 'uname -m; cat /host/from-host.txt; echo written > /host/from-vm.txt'
    const outputs = await run(agent, `shell ${JSON.stringify({ cmd })}`)
    const said = JSON.stringify(outputs[1]?.message.contents)
    console.log(`  vm: ${said}`)
    check(said.includes('by volume'), "the agent's shell tool reads the sandbox's volume")
    check((await volume.readFile('from-vm.txt')).toString('utf8').trim() === 'written', "the volume keeps the sandbox's write")
    await agent.close()
    await sandbox.remove()
    await volume.remove()
    await client.close()
  }
} finally {
  await model.close()
}

console.log(failures.length ? `FAILED: ${failures.join('; ')}` : 'ALL PASS')
process.exit(failures.length ? 1 : 0)
