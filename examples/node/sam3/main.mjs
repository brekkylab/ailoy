// Segment images and videos with SAM3 through ncnn on the guest's Vulkan device, as an agent's
// skill.
//
//     node main.mjs
//     node main.mjs "Find every cat in the photos in context and mask them"
//
// The Node side of the Rust `sam3` example, whose `main.rs` has the long form. The skill
// (`SKILL.md`, `run_sam3.py`, mounted from memory at `/skills/sam3`) and `prepare_model.py` are
// the ones the three sides share, in `examples/shared/sam3`. What a run uses is in this folder:
// `data/`, which `prepare_model.py` downloads and converts the model into on the first run, and
// these two:
//
// * `context/` at `/context`, read-only — the images and frames to segment, when they are not
//   in the prompt.
// * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//
// Environment, also read from `.env`:
//
// * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
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
const SHARED = resolve(HERE, '../../shared/sam3')
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
      .withFile('run_sam3.py', readFileSync(join(SHARED, 'run_sam3.py'))),
    join(PROJECT, 'skill'),
  )

  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()

  let consoleClient, agent
  try {
    // `build()` builds the image before it starts the console, which takes a while.
    console.log('building the image ...')
    consoleClient = await ConsoleClient.builder()
      // Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
      .image(
        new Recipe('python:3.12-slim-trixie')
          // `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the wheel
          // opens. Mesa from backports: venus passes VK_KHR_shader_bfloat16 and
          // VK_KHR_cooperative_matrix through from 26.0 on, and trixie itself has 25.0.
          .step(
            "echo 'deb http://deb.debian.org/debian trixie-backports main' " +
              '> /etc/apt/sources.list.d/backports.list ' +
              '&& apt-get update && apt-get install -y --no-install-recommends ' +
              '-t trixie-backports mesa-vulkan-drivers ' +
              '&& apt-get install -y --no-install-recommends libvulkan1 ' +
              '&& rm -rf /var/lib/apt/lists/*',
          )
          // Headless OpenCV: `opencv-python` needs libGL, which the slim image lacks. `ncnn` asks
          // for it by name, and both wheels install the same `cv2` package, so `--no-deps` on
          // `ncnn` and its own dependencies spelled out, opencv aside.
          .step(
            'pip install --no-cache-dir av numpy opencv-python-headless pillow ' +
              'portalocker requests tokenizers tqdm ' +
              '&& pip install --no-cache-dir --no-deps ncnn',
          ),
      )
      .mountReadonly(join(PROJECT, 'data', 'ncnn'), '/models')
      .mountReadonly(skill, '/skills/sam3')
      .mountReadonly(join(PROJECT, 'context'), '/context')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      .gpu(true)
      .vcpus(2)
      .memoryMib(4096)
      .gpuMemoryMib(12288)
      .build()

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'openai/gpt-6-astra')
      .instruction(INSTRUCTION)
      .systemTools()
      .webFetchTool()
      .webSearchTool([])
      .console(consoleClient)
      .skill('/skills/sam3')
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
      // A run prints a summary of what it found and where the masks went, and ncnn's device
      // log: shown whole.
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
