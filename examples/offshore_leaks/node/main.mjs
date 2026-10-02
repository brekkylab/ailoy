// Analyze ICIJ's Offshore Leaks database with an agent that writes and runs its own SQL and
// Python, in a console, against the data it is given as context.
//
//     node main.mjs
//     node main.mjs "Which South Korean officers appear in more than one leak?"
//     node main.mjs "Map who is behind the entities Mossack Fonseca set up in Niue"
//
// The Node side of the Rust `offshore_leaks` example, whose `main.rs` has the long form. The
// skill (`SKILL.md`, `oldb.py`, mounted from memory at `/skills/offshore-leaks`) and
// `prepare_data.py` are the ones the three sides share, in `examples/offshore_leaks/shared`.
// What a run uses is in this folder: `data/`, where `prepare_data.py` downloads ICIJ's CSVs on
// the first run before loading them into one DuckDB file in `context/`, and these two:
//
// * `context/` at `/context`, read-only — `offshore_leaks.duckdb`, and whatever else the
//   request is about, such as a list of names to look for.
// * `artifacts/` at `/artifacts`, writable — where the reports, tables and charts go.
//
// The agent's code runs in the console, which sees nothing of the host but these two
// directories, and has no network.
//
// The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA. Being
// in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.
//
// Environment, also read from `.env`:
//
// * `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads; ICIJ's latest by default.
// * `UV` — the `uv` binary `prepare_data.py` runs with; `uv` on `PATH` by default.
// * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's API
//   key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import ailoy from 'ailoy-node'

const HERE = dirname(fileURLToPath(import.meta.url))
// Absolute, because a mount is named to the server as a `file://` URL.
// What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE

const { AgentBuilder, ConsoleClient, Directory, HostMount, Recipe, ensureVirtx } = ailoy

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
  // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
  for (const name of ['context', 'artifacts', 'skill']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }
  await prepare(SHARED, PROJECT)

  // Held here, not left to garbage collection: Node runs no finalizer on exit, and a mount
  // comes down only when the last holder lets go.
  const skill = new HostMount(
    new Directory()
      .withFile('SKILL.md', readFileSync(join(SHARED, 'SKILL.md')))
      .withFile('oldb.py', readFileSync(join(SHARED, 'oldb.py'))),
    join(PROJECT, 'skill'),
  )

  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()

  let consoleClient, agent
  try {
    consoleClient = await ConsoleClient.builder()
      .image(
        new Recipe('python:3.12-slim-trixie').step(
          // The DuckDB `prepare_data.py` writes the file with (pinned in pyproject.toml); an
          // older one may not read it.
          'pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx',
        ),
      )
      .mountReadonly(skill, '/skills/offshore-leaks')
      .mountReadonly(join(PROJECT, 'context'), '/context')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      .network(false)
      .vcpus(2)
      .memoryMib(2048)
      .build()

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'openai/gpt-6-astra')
      .instruction(INSTRUCTION)
      .maxTokens(64000)
      .systemTools()
      .console(consoleClient)
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
  } finally {
    await agent?.close()
    await consoleClient?.close()
    await skill.unmount()
  }
}

loadDotenv()
await main(process.argv.slice(2).join(' ') || QUERY)
