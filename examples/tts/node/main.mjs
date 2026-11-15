// Speak a text in a voice described in words with Qwen3-TTS, through ncnn on the guest's Vulkan
// device, as an agent's skill.
//
//     node main.mjs
//
// The Node side of the Rust `tts` example, whose `main.rs` has the long form. The skill
// (`SKILL.md`, `run_tts.py`, put each run onto a volume of the virtx server, `ailoy-tts-skill`,
// mounted at `/skills/tts`), `prepare_model.py` and `context_example/` are the ones the three
// sides share, in `examples/tts/shared`. What a run uses is in this folder: `data/`, which
// `prepare_model.py` downloads and converts the models into on the first run, `context/` and
// `artifacts/`, each on a volume of its own:
//
// * The volume `ailoy-tts-models` at `/models`, read-only — `data/ncnn/`, put there each run.
// * The volume `ailoy-tts-context` at `/context`, read-only — `context/`, the text and the
//   instruction, `text.txt` and `instruct.txt`, put there each run; `context_example/`'s, when
//   it is empty.
// * The volume `ailoy-tts-artifacts` at `/artifacts`, writable — where what the agent hands back
//   goes. It stays on the server between runs, and is copied to `artifacts/` here at the end.
//
// There is no prompt: the agent reads both files, speaks the text in that voice and hands back
// the WAV.
//
// Qwen3-TTS and its weights are the Qwen team's, under Apache-2.0.
//
// Environment, also read from `.env`:
//
// * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
// * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//   has to be set up as `docs/content/guides/model-providers.mdx` says.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
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
const IMAGE = 'ailoy-tts:1'

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

const QUERY =
  'Speak the text in /context in the voice and manner its instruction describes, ' +
  'and hand back the speech as a WAV file.'

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

// Download and convert the models into `project/data`.
async function prepare(shared, project) {
  const uv = process.env.UV ?? 'uv'
  // An activated environment elsewhere is not this project's, and uv says so.
  const { VIRTUAL_ENV, ...env } = process.env
  const child = spawn(uv, ['run', 'prepare_model.py', join(project, 'data')], {
    cwd: shared,
    env,
    stdio: 'inherit',
  })
  const [code, signal] = await once(child, 'exit').catch((e) => {
    throw new Error(`running \`${uv}\`. Install uv, or point $UV at it.`, { cause: e })
  })
  if (code !== 0) throw new Error(`preparing the models: ${signal ?? `exit status ${code}`}`)
}

async function main() {
  await prepare(SHARED, PROJECT)
  for (const name of ['context', 'artifacts']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }
  // Something to say, for a first run.
  const context = join(PROJECT, 'context')
  if (readdirSync(context).length === 0) {
    cpSync(join(SHARED, 'context_example'), context, { recursive: true })
  }

  // The virtx server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none. The sandboxes it holds go when it is closed.
  const client = Client.spawn()

  let agent
  try {
    // The image, built once. Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
    // The server answers a tag in its full form, as `docker.io/library/ailoy-tts:1`.
    if (!(await client.listImage()).some((image) => image.reference.split('/').pop() === IMAGE)) {
      console.log(`Building ${IMAGE}, once; it takes a few minutes.`)
      const recipe = new Recipe('python:3.12-slim-trixie')
        // `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the wheel
        // opens. Mesa from backports: 26.0 against trixie's 25.0.
        .run(
          "echo 'deb http://deb.debian.org/debian trixie-backports main' " +
            '> /etc/apt/sources.list.d/backports.list ' +
            '&& apt-get update && apt-get install -y --no-install-recommends ' +
            '-t trixie-backports mesa-vulkan-drivers ' +
            '&& apt-get install -y --no-install-recommends libvulkan1 ' +
            '&& rm -rf /var/lib/apt/lists/*',
        )
        .run('pip install --no-cache-dir ncnn numpy tokenizers')
      for await (const entry of client.buildImage(recipe, IMAGE)) {
        if ('end' in entry && entry.end.error) throw new Error(`building ${IMAGE}: ${entry.end.error}`)
      }
    }

    // The volumes, made the first time and kept on the server: `409` says one is there already.
    const volumes = []
    for (const name of ['ailoy-tts-models', 'ailoy-tts-skill', 'ailoy-tts-context', 'ailoy-tts-artifacts']) {
      volumes.push(await client.createVolume(name).catch((e) => {
        if (e.status !== 409) throw e
        return client.getVolume(name)
      }))
    }
    const [models, skill, contextVolume, artifacts] = volumes

    // `data/ncnn/` and `context/`, every file of them, onto theirs.
    for (const [source, volume] of [
      [join(PROJECT, 'data', 'ncnn'), models],
      [context, contextVolume],
    ]) {
      for (const to of readdirSync(source, { recursive: true }).sort()) {
        const path = join(source, to)
        if (statSync(path).isDirectory()) await volume.createDirAll(to)
        else await volume.writeFile(to, readFileSync(path))
      }
    }
    // The skill, from `shared/`.
    for (const name of ['SKILL.md', 'run_tts.py']) {
      await skill.writeFile(name, readFileSync(join(SHARED, name)))
    }

    const sandbox = await client.createSandbox(IMAGE, {
      volumes: [
        { volume: models, path: '/models', readonly: true },
        { volume: skill, path: '/skills/tts', readonly: true },
        { volume: contextVolume, path: '/context', readonly: true },
        { volume: artifacts, path: '/artifacts' },
      ],
      gpu: true,
      vcpus: 2,
      // The talker is 2.8 GB on the device in fp16, and its cache and the codec's buffers come
      // on top of it there; the run itself takes under 1 GB of memory.
      memoryMib: 4096,
      gpuMemoryMib: 8192,
    })

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5')
      .instruction(INSTRUCTION)
      .sandbox(sandbox)
      .skill('/skills/tts')
      .build()

    for await (const { message } of agent.run(QUERY)) {
      if (message.role === 'assistant') {
        for (const part of message.contents) {
          if (part.type === 'text') console.log(part.text)
        }
        for (const call of message.tool_calls ?? []) {
          console.log(`→ ${call.function.name} ${pretty(call.function.arguments)}`)
        }
      }
      // A run prints where the speech went and how long it is, and ncnn's device log and its
      // progress: shown whole.
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
  } finally {
    await agent?.close()
    await client.close()
  }
}

loadDotenv()
await main()
