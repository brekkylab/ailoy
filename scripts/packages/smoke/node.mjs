// The Node package as a user installs it, on the platform this runs on.
//
// Run from a project that installed `@brekkylab/ailoy`, so it resolves the way that
// project's code would. What this platform can do is said in the environment:
//   SMOKE_MOUNT   1 if a FUSE provider is installed here, else 0
//   SMOKE_SERVER  1 if the console server is in $VIRTX_HOME/bin, else 0
//   SMOKE_VM      1 if this machine can boot one (KVM or HVF), else 0
//
// A turn's model is served from this process (`fake.mjs`): no network, and no API key.
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
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
check(typeof ailoy.AgentBuilder === 'function', 'the package loads')

// Only this platform's binary, as npm's os/cpu/libc filter chose it.
const scope = path.join(process.cwd(), 'node_modules', '@brekkylab')
const installed = fs.readdirSync(scope).sort()
console.log(`  installed: ${installed.join(' ')}`)
check(installed.length === 2, 'exactly one platform package was installed beside the root')

check(new ailoy.Recipe('alpine:latest').step('echo hi').toString().includes('echo hi'), 'virtx comes built in')

const model = await fakeModel()
try {
  ailoy.registerLangModel('fake/*', 'chat_completion', model.url)

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

  // Whether or not this host can mount: without a provider, a mount is an error, not a crash.
  const point = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoy-smoke-'))
  let mount = null
  try {
    mount = new ailoy.HostMount(new ailoy.Directory().withFile('a.txt', 'hi'), point)
  } catch (e) {
    console.log(`  HostMount: ${e.message}`)
  }
  check(!!mount === want('SMOKE_MOUNT'), `a HostMount is ${mount ? '' : 'not '}made`)
  if (mount) {
    check(fs.readFileSync(path.join(point, 'a.txt'), 'utf8') === 'hi', 'HostMount serves its tree')
    await mount.unmount()
    check(!fs.existsSync(path.join(point, 'a.txt')), 'HostMount.unmount() takes it down')
  }

  if (want('SMOKE_SERVER')) {
    // The server runs on this machine, and answers -- no VM needed to ask its version.
    const images = await ailoy.ImageClient.tryNew()
    const version = await images.version()
    await images.close()
    check(typeof version === 'string' && version.length > 0, `the console server answers (protocol ${version})`)
  }

  if (want('SMOKE_VM')) {
    // An agent's shell tool, in a VM session that sees a host directory.
    const host = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoy-smoke-host-'))
    fs.writeFileSync(path.join(host, 'from-host.txt'), 'by path')
    const console_ = await ailoy.ConsoleClient.builder().image(new ailoy.Recipe('alpine:latest')).mount(host, '/host').build()
    const agent = await new ailoy.AgentBuilder('fake/model').shellTool().console(console_).build()
    const cmd = 'uname -m; cat /host/from-host.txt; echo written > /host/from-vm.txt'
    const outputs = await run(agent, `shell ${JSON.stringify({ cmd })}`)
    const said = JSON.stringify(outputs[1]?.message.contents)
    console.log(`  vm: ${said}`)
    check(said.includes('by path'), "the agent's shell tool reads the session's mount")
    check(fs.readFileSync(path.join(host, 'from-vm.txt'), 'utf8').trim() === 'written', "the host sees the VM's write")
    await agent.close()
    await console_.close()
  }
} finally {
  await model.close()
}

console.log(failures.length ? `FAILED: ${failures.join('; ')}` : 'ALL PASS')
process.exit(failures.length ? 1 : 0)
