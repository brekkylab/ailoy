// Answer typed questions with Laya through ncnn on the guest's Vulkan device, as an agent's
// skill.
//
//     node main.mjs
//     node main.mjs "Route the tickets in /context/inbox, but send anything about contracts to legal"
//
// The Node side of the Rust `laya` example, whose `main.rs` has the long form. The skill
// (`SKILL.md`, `run_laya.py`, put each run onto the volume `ailoy-laya-skill` of the virtx server
// mounted at `/skills/laya`) and `prepare_model.py` are the ones the three sides share, in
// `examples/laya/shared`. What a run uses is in this folder: `data/`, which `prepare_model.py`
// downloads and converts the model into on the first run and whose `ncnn/` goes each run onto
// the volume `ailoy-laya-models` at `/models`, read-only, and these two:
//
// * The volume `ailoy-laya-context` at `/context`, read-only — `context/`, what to decide on,
//   when it is not in the prompt, put there each run. `context/inbox/` is a fresh copy of
//   `shared/tickets/` on every run.
// * The volume `ailoy-laya-artifacts` at `/artifacts`, writable — where what the agent hands back
//   goes: the tickets filed under `routes/<department>/`, emptied at the start of every run. It
//   stays on the server between runs, and is copied to `artifacts/` here at the end.
//
// Without a prompt the agent routes a support inbox: it asks Laya which department each ticket in
// `shared/tickets/` belongs to and files a copy in that department's folder.
//
// Environment, also read from `.env`:
//
// * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
// * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//   has to be set up as `docs/src/content/docs/guide/model-providers.mdx` says.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'
import virtx from '@brekkylab/virtx'

const HERE = dirname(fileURLToPath(import.meta.url))
// What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE

const { AgentBuilder } = ailoy
const { Client, Recipe } = virtx

// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE = 'ailoy-laya:1'

// The request when none is given.
const QUERY =
  'Route every ticket in /context/inbox to the department that should handle it, by what the ' +
  'customer needs rather than the words they use: engineering, finance, sales, legal or ' +
  'marketing. Decide each one with Laya, copy the ticket into /artifacts/routes/<department>/, ' +
  'and write what went where to /artifacts/routing.md.'

const INSTRUCTION =
  '# Context\n\n' +
  'Path: `/context`\n\n' +
  'This folder holds the data and context the user wants to share with you. ' +
  'When the user refers to something whose context you cannot figure out, the files in this folder might help. ' +
  'The information that settles the answer may be here too, and so may hints toward it, so look through this folder for them.\n\n' +
  '# Artifacts\n\n' +
  'Path: `/artifacts`\n\n' +
  'This folder is where what the user asked for goes. ' +
  'Write every result here, such as a report, a figure, or the file the user came for. ' +
  'Everything in this folder is collected and handed back to the user, and a result left anywhere else is not delivered.'

// From the nearest `.env` up from this file. What the environment already has wins.
function loadDotenv() {
  for (let dir = HERE; ; dir = dirname(dir)) {
    const path = join(dir, '.env')
    if (existsSync(path)) return process.loadEnvFile(path)
    if (dirname(dir) === dir) return
  }
}

// `JSON.stringify` with an image's bytes as their length, not a list of every one of them.
function pretty(value) {
  return JSON.stringify(
    value,
    function (key, json) {
      const raw = this[key]
      return Buffer.isBuffer(raw) ? `<${raw.length} bytes>` : json
    },
    2,
  )
}

// Download and convert the model into `project/data`.
async function prepare(shared, project) {
  const uv = process.env.UV ?? 'uv'
  // An activated environment elsewhere is not this project's, and uv says so.
  const { VIRTUAL_ENV, ...env } = process.env
  // `transformers` probes for TensorFlow at import, which can hang model construction.
  env.USE_TF = '0'
  const child = spawn(uv, ['run', 'prepare_model.py', join(project, 'data')], {
    cwd: shared,
    env,
    stdio: 'inherit',
  })
  const [code, signal] = await once(child, 'exit').catch((e) => {
    throw new Error(`running \`${uv}\`. Install uv, or point $UV at it.`, { cause: e })
  })
  if (code !== 0) throw new Error(`preparing the model: ${signal ?? `exit status ${code}`}`)
}

// Reset `project/context/inbox` to a copy of `shared/tickets`, so every run routes the same inbox.
function fillInbox(shared, project) {
  const inbox = join(project, 'context', 'inbox')
  rmSync(inbox, { recursive: true, force: true })
  mkdirSync(inbox, { recursive: true })
  const tickets = join(shared, 'tickets')
  for (const entry of readdirSync(tickets, { withFileTypes: true })) {
    if (entry.isFile()) copyFileSync(join(tickets, entry.name), join(inbox, entry.name))
  }
}

// Empty `project/artifacts/routes`, so what is filed there is only this run's.
function clearRoutes(project) {
  rmSync(join(project, 'artifacts', 'routes'), { recursive: true, force: true })
}

async function main(prompt) {
  await prepare(SHARED, PROJECT)
  for (const name of ['context', 'artifacts']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }
  fillInbox(SHARED, PROJECT)
  clearRoutes(PROJECT)

  // The virtx server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none. The sandboxes it holds go when it is closed.
  const client = Client.spawn()

  // The image, built once. The server answers a tag in its full form, as
  // `docker.io/library/ailoy-laya:1`.
  if (!(await client.listImage()).some((image) => image.reference.split('/').pop() === IMAGE)) {
    console.log(`Building ${IMAGE}, once; it takes a few minutes.`)
    // Debian, not Alpine: PyPI's musllinux ncnn wheel crashes freeing its first `Mat`, where the
    // manylinux (glibc) one runs laya.
    const recipe = new Recipe('python:3.12-slim-trixie')
      .run('pip install --no-cache-dir ncnn numpy tokenizers')
    for await (const entry of client.buildImage(recipe, IMAGE)) {
      if ('end' in entry && entry.end.error) throw new Error(`building ${IMAGE}: ${entry.end.error}`)
    }
  }

  // The volumes, made the first time and kept on the server: `409` says one is there already.
  const volumes = []
  for (const name of ['ailoy-laya-models', 'ailoy-laya-skill', 'ailoy-laya-context', 'ailoy-laya-artifacts']) {
    volumes.push(await client.createVolume(name).catch((e) => {
      if (e.status !== 409) throw e
      return client.getVolume(name)
    }))
  }
  const [models, skill, context, artifacts] = volumes

  // The skill, `SKILL.md` and `run_laya.py` of `shared/`, onto its own.
  for (const name of ['SKILL.md', 'run_laya.py']) {
    await skill.writeFile(name, readFileSync(join(SHARED, name)))
  }
  // A fresh inbox and no routes, as on this side.
  await context.removeFile('inbox', true)
  await artifacts.removeFile('routes', true)
  // `data/ncnn/` and `context/`, every file of them, onto theirs.
  for (const [volume, root] of [
    [models, join(PROJECT, 'data', 'ncnn')],
    [context, join(PROJECT, 'context')],
  ]) {
    for (const to of readdirSync(root, { recursive: true }).sort()) {
      const path = join(root, to)
      if (statSync(path).isDirectory()) await volume.createDirAll(to)
      else await volume.writeFile(to, readFileSync(path))
    }
  }

  const sandbox = await client.createSandbox(IMAGE, {
    volumes: [
      { volume: models, path: '/models', readonly: true },
      { volume: skill, path: '/skills/laya', readonly: true },
      { volume: context, path: '/context', readonly: true },
      { volume: artifacts, path: '/artifacts' },
    ],
    vcpus: 2,
    memoryMib: 4096,
    gpu: true,
    gpuMemoryMib: 12288,
  })

  const agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5')
    .instruction(INSTRUCTION)
    .webFetchTool()
    .webSearchTool([])
    .sandbox(sandbox)
    .skill('/skills/laya')
    .build()

  for await (const { message } of agent.run(prompt)) {
    if (message.role === 'assistant') {
      for (const part of message.contents) {
        if (part.type === 'text') console.log(part.text)
      }
      for (const call of message.tool_calls ?? []) {
        console.log(`→ ${call.function.name} ${pretty(call.function.arguments)}`)
      }
    }
    // Laya's answers are small, and they are what the reply is made from: shown whole.
    else if (message.role === 'tool') {
      for (const part of message.contents) console.log(`← ${pretty(part)}`)
    }
  }

  // The artifacts, every file of them, to `artifacts/` here.
  const dirs = ['']
  while (dirs.length) {
    const at = dirs.pop()
    mkdirSync(join(PROJECT, 'artifacts', at), { recursive: true })
    for (const entry of await artifacts.readDir(at)) {
      if (entry.metadata.isDir) dirs.push(`${at}${entry.name}/`)
      else writeFileSync(join(PROJECT, 'artifacts', at, entry.name), await artifacts.readFile(`${at}${entry.name}`))
    }
  }

  await agent.close()
  await client.close()
}

loadDotenv()
await main(process.argv.slice(2).join(' ') || QUERY)
