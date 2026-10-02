// Answer typed questions with Laya through ncnn on the guest's Vulkan device, as an agent's
// skill.
//
//     node main.mjs
//     node main.mjs "Triage this ticket: we were billed twice for March ..."
//
// The Node side of the Rust `laya` example, whose `main.rs` has the long form. The skill
// (`SKILL.md`, `run_laya.py`, mounted from memory at `/skills/laya`) and `prepare_model.py` are
// the ones the three sides share, in `examples/laya/shared`. What a run uses is in this folder:
// `data/`, which `prepare_model.py` downloads and converts the model into on the first run, and
// these two:
//
// * `context/` at `/context`, read-only — what to decide on, when it is not in the prompt.
// * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//
// Environment, also read from `.env`:
//
// * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
// * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//   API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
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

async function main(prompt) {
  await prepare(SHARED, PROJECT)
  // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
  for (const name of ['context', 'artifacts', 'skill']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }

  // Held here, not left to garbage collection: Node runs no finalizer on exit, and a mount
  // comes down only when the last holder lets go.
  const skill = new HostMount(
    new Directory()
      .withFile('SKILL.md', readFileSync(join(SHARED, 'SKILL.md')))
      .withFile('run_laya.py', readFileSync(join(SHARED, 'run_laya.py'))),
    join(PROJECT, 'skill'),
  )

  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()

  let consoleClient, agent
  try {
    consoleClient = await ConsoleClient.builder()
      // Debian, not Alpine: PyPI's musllinux ncnn wheel crashes freeing its first `Mat`, where
      // the manylinux (glibc) one runs laya.
      .image(
        new Recipe('python:3.12-slim-trixie')
          // `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the wheel
          // opens. Mesa from backports: venus passes VK_KHR_shader_bfloat16 through from 26.0
          // on, and trixie itself has 25.0.
          .step(
            "echo 'deb http://deb.debian.org/debian trixie-backports main' " +
              '> /etc/apt/sources.list.d/backports.list ' +
              '&& apt-get update && apt-get install -y --no-install-recommends ' +
              '-t trixie-backports mesa-vulkan-drivers ' +
              '&& apt-get install -y --no-install-recommends libvulkan1 ' +
              '&& rm -rf /var/lib/apt/lists/*',
          )
          .step('pip install --no-cache-dir ncnn numpy tokenizers'),
      )
      .mountReadonly(join(PROJECT, 'data', 'ncnn'), '/models')
      .mountReadonly(skill, '/skills/laya')
      .mountReadonly(join(PROJECT, 'context'), '/context')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      .gpu(true)
      .vcpus(2)
      .memoryMib(4096)
      .gpuMemoryMib(12288)
      .build()

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5')
      .instruction(INSTRUCTION)
      .maxTokens(64000)
      .systemTools()
      .webFetchTool()
      .webSearchTool([])
      .console(consoleClient)
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
  } finally {
    await agent?.close()
    await consoleClient?.close()
    await skill.unmount()
  }
}

loadDotenv()
await main(process.argv.slice(2).join(' '))
