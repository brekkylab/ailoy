// Analyze ICIJ's Offshore Leaks database with an agent that writes and runs its own SQL and
// Python, in a sandbox, against the data it is given as context.
//
//     node main.mjs
//     node main.mjs "Which South Korean officers appear in more than one leak?"
//     node main.mjs "Map who is behind the entities Mossack Fonseca set up in Niue"
//
// The Node side of the Rust `offshore_leaks` example, whose `main.rs` has the long form. The
// skill (`SKILL.md`, `oldb.py`, which go each run onto the volume `ailoy-offshore-leaks-skill`
// of the virtx server, mounted at `/skills/offshore-leaks`) and `prepare_data.py` are the ones
// the three sides share, in `examples/offshore_leaks/shared`. What a run uses is in this
// folder: `data/`, where `prepare_data.py` downloads ICIJ's CSVs on the first run before
// loading them into one DuckDB file in `context/`, and these two volumes:
//
// * `ailoy-offshore-leaks-context` at `/context`, read-only — `context/` here, every file of
//   it, put there each run: `offshore_leaks.duckdb`, and whatever else the request is about,
//   such as a list of names to look for.
// * `ailoy-offshore-leaks-artifacts` at `/artifacts`, writable — where the reports, tables and
//   charts go. It stays on the server between runs, and is copied to `artifacts/` here at the
//   end.
//
// The agent's code runs in the sandbox, which sees nothing of the host but these three
// volumes, and has no network.
//
// The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA. Being
// in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.
//
// Environment, also read from `.env`:
//
// * `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads; ICIJ's latest by default.
// * `UV` — the `uv` binary `prepare_data.py` runs with; `uv` on `PATH` by default.
// * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to be
//   set up as `docs/content/guides/model-providers.mdx` says.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
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
const IMAGE = 'ailoy-offshore-leaks:1'

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

// The request when none is given.
const QUERY =
  'Which intermediaries set up the most entities in the Panama Papers, and in which ' +
  'jurisdictions? Write a short report with a chart.'

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

// Download the database and load it into `project/context`.
async function prepare(shared, project) {
  const uv = process.env.UV ?? 'uv'
  // An activated environment elsewhere is not this project's, and uv says so.
  const { VIRTUAL_ENV, ...env } = process.env
  const data = join(project, 'data')
  const database = join(project, 'context', 'offshore_leaks.duckdb')
  const child = spawn(uv, ['run', 'prepare_data.py', data, database], {
    cwd: shared,
    env,
    stdio: 'inherit',
  })
  const [code, signal] = await once(child, 'exit').catch((e) => {
    throw new Error(`running \`${uv}\`. Install uv, or point $UV at it.`, { cause: e })
  })
  if (code !== 0) throw new Error(`preparing the data: ${signal ?? `exit status ${code}`}`)
}

async function main(prompt) {
  for (const name of ['context', 'artifacts']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }
  await prepare(SHARED, PROJECT)

  // The virtx server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none. The sandboxes it holds go when it is closed.
  const client = Client.spawn()

  // The image, built once: Python, and what the agent's code reads the data and draws with.
  // The server answers a tag in its full form, as `docker.io/library/ailoy-offshore-leaks:1`.
  if (!(await client.listImage()).some((image) => image.reference.split('/').pop() === IMAGE)) {
    console.log(`Building ${IMAGE}, once; it takes a few minutes.`)
    const recipe = new Recipe('python:3.12-slim-trixie').run(
      // The DuckDB `prepare_data.py` writes the file with (pinned in pyproject.toml); an
      // older one may not read it.
      'pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx',
    )
    for await (const entry of client.buildImage(recipe, IMAGE)) {
      if ('end' in entry && entry.end.error) throw new Error(`building ${IMAGE}: ${entry.end.error}`)
    }
  }

  // The volumes, made the first time and kept on the server: `409` says one is there already.
  const volumes = []
  for (const name of ['ailoy-offshore-leaks-skill', 'ailoy-offshore-leaks-context', 'ailoy-offshore-leaks-artifacts']) {
    volumes.push(await client.createVolume(name).catch((e) => {
      if (e.status !== 409) throw e
      return client.getVolume(name)
    }))
  }
  const [skill, context, artifacts] = volumes

  // The skill onto the first.
  for (const name of ['SKILL.md', 'oldb.py']) {
    await skill.writeFile(name, readFileSync(join(SHARED, name)))
  }

  // `context/`, every file of it, onto the second.
  for (const to of readdirSync(join(PROJECT, 'context'), { recursive: true }).sort()) {
    const path = join(PROJECT, 'context', to)
    if (statSync(path).isDirectory()) await context.createDirAll(to)
    else await context.writeFile(to, readFileSync(path))
  }

  const sandbox = await client.createSandbox(IMAGE, {
    volumes: [
      { volume: skill, path: '/skills/offshore-leaks', readonly: true },
      { volume: context, path: '/context', readonly: true },
      { volume: artifacts, path: '/artifacts' },
    ],
    network: false,
    vcpus: 2,
    memoryMib: 2048,
  })

  const agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'openai/gpt-6-astra')
    .instruction(INSTRUCTION)
    .sandbox(sandbox)
    .skill('/skills/offshore-leaks')
    .build()

  for await (const { message } of agent.run(prompt)) {
    if (message.role === 'assistant') {
      for (const part of message.contents) {
        if (part.type === 'text') console.log(part.text)
      }
      for (const call of message.tool_calls ?? []) {
        console.log(`→ ${call.function.name} ${pretty(call.function.arguments)}`)
      }
    } else if (message.role === 'tool') {
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
