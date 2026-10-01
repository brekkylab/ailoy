// Design 3D parts with an agent that writes CadQuery, looks at what it built and fixes it, in a
// console.
//
//     node examples/cad/main.mjs
//     node examples/cad/main.mjs "A wall mount for a 60 mm fan, with four M3 screw holes"
//     node examples/cad/main.mjs "A twisted vase, 150 mm tall, hexagonal at the base and round at the top"
//
// The Node side of the Rust `cad` example, whose `main.rs` has the long form. It runs on that
// example's folder, `examples/cad` at the top of the checkout: the skill (`SKILL.md`,
// `render.py`, mounted from memory at `/skills/cad`), and these two:
//
// * `context/` at `/context`, read-only — what the request is about, when it is not in the
//   prompt: a sketch, a photo of the thing it has to fit, the STEP of a part to mate with.
// * `artifacts/` at `/artifacts`, writable — the script, the STEP, STL and GLB files, and the
//   pictures.
//
// `render.py` rasterizes on the CPU, so there is no model to download and no GPU needed.
//
// Environment, also read from `.env`:
//
// * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's API
//   key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...). It has to take images, or
//   it cannot see what it built.
//
// The addon has to be built first (`npm run build` in `bindings/node`).

import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
// Absolute, because a mount is named to the server as a `file://` URL.
const PROJECT = resolve(HERE, '../../../../examples/cad')

const { AgentBuilder, ConsoleClient, Directory, HostMount, Recipe } = createRequire(
  import.meta.url,
)('../../index.js')

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
  'Design a gear bearing that prints in one piece, already assembled: a sun, five planets and ' +
  'a ring, all with herringbone teeth so the planets cannot slide out, and enough clearance ' +
  'that it turns when it comes off the bed. About 60 mm across and 15 mm tall, with a ' +
  'hexagonal hole through the sun for a key. Show it assembled, cut in half and turning.'

// From the nearest `.env` up from this file. What the environment already has wins.
function loadDotenv() {
  for (let dir = HERE; ; dir = dirname(dir)) {
    const path = join(dir, '.env')
    if (existsSync(path)) return process.loadEnvFile(path)
    if (dirname(dir) === dir) return
  }
}

const pretty = (value) => JSON.stringify(value, null, 2)

async function main(prompt) {
  // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
  for (const name of ['context', 'artifacts', 'skill']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }

  // Held here, not left to garbage collection: Node runs no finalizer on exit, and a mount
  // comes down only when the last holder lets go.
  const skill = new HostMount(
    new Directory()
      .withFile('SKILL.md', readFileSync(join(PROJECT, 'SKILL.md')))
      .withFile('render.py', readFileSync(join(PROJECT, 'render.py'))),
    join(PROJECT, 'skill'),
  )

  let consoleClient, agent
  try {
    consoleClient = await ConsoleClient.builder()
      .image(
        new Recipe('python:3.12-slim-trixie')
          // OpenCascade's wheel links libGL and libX11, which the slim image leaves out.
          .step(
            'apt-get update && apt-get install -y --no-install-recommends ' +
              'libgl1 libx11-6 && rm -rf /var/lib/apt/lists/*',
          )
          .step('pip install --no-cache-dir vtk==9.6.2')
          .step('pip install --no-cache-dir cadquery-ocp==7.9.3.1.1')
          .step(
            'pip install --no-cache-dir --no-deps cadquery==2.8.0 ' +
              '&& pip install --no-cache-dir casadi ezdxf multimethod nlopt pyparsing ' +
              'runtype scipy typing_extensions trimesh numpy pillow',
          ),
      )
      .mountReadonly(skill, '/skills/cad')
      .mountReadonly(join(PROJECT, 'context'), '/context')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      .vcpus(4)
      .memoryMib(4096)
      .build()

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'openai/gpt-6-astra')
      .instruction(INSTRUCTION)
      // A whole model script is one `write`, and the model thinks before it, which counts
      // against the same limit: far more than the 8192 tokens a reply gets by default.
      .maxTokens(64000)
      .systemTools()
      .console(consoleClient)
      .skill('/skills/cad')
      .build()

    for await (const { message, finish_reason } of agent.run(prompt)) {
      // A token-limit cutoff also ends the run; without this it ends silently.
      if (finish_reason.type === 'length') {
        console.error('(the reply was cut off at the token limit)')
      }
      if (message.role === 'assistant') {
        for (const part of message.contents) {
          if (part.type === 'text') console.log(part.text)
        }
        for (const call of message.tool_calls ?? []) {
          console.log(`→ ${call.function.name} ${pretty(call.function.arguments)}`)
        }
      }
      // A `read` of a picture is an image part: its bytes are no use on a terminal.
      else if (message.role === 'tool') {
        for (const part of message.contents) {
          console.log(part.type === 'image' ? '← [image]' : `← ${pretty(part)}`)
        }
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
