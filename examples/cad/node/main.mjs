// Design 3D parts with an agent that writes CadQuery, looks at what it built and fixes it, in a
// sandbox.
//
//     node main.mjs
//     node main.mjs "A wall mount for a 60 mm fan, with four M3 screw holes"
//     node main.mjs "A twisted vase, 150 mm tall, hexagonal at the base and round at the top"
//
// The Node side of the Rust `cad` example, whose `main.rs` has the long form. The skill
// (`SKILL.md`, `render.py`) is the one the three sides share, in `examples/cad/shared`, put
// each run onto the volume `ailoy-cad-skill` of the virtx server mounted at `/skills/cad`. What
// a run uses is in this folder:
//
// * The volume `ailoy-cad-context` at `/context`, read-only — `context/` here, put there each
//   run: what the request is about, when it is not in the prompt: a sketch, a photo of the
//   thing it has to fit, the STEP of a part to mate with.
// * The volume `ailoy-cad-artifacts` at `/artifacts`, writable — the script, the STEP, STL and
//   GLB files, and the pictures. It stays on the server between runs, and is copied to
//   `artifacts/` here at the end.
//
// `render.py` rasterizes on the CPU, so there is no model to download and no GPU needed.
//
// Environment, also read from `.env`:
//
// * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to be
//   set up as `docs/src/content/docs/guide/model-providers.mdx` says. It has to take images,
//   or it cannot see what it built.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'
import virtx from '@brekkylab/virtx'

const HERE = dirname(fileURLToPath(import.meta.url))
// What the Rust, Python and Node sides share: the skill.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE

const { AgentBuilder } = ailoy
const { Client, Recipe } = virtx

// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE = 'ailoy-cad:1'

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
  mkdirSync(join(PROJECT, 'context'), { recursive: true })

  // The virtx server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none. The sandboxes it holds go when it is closed.
  const client = Client.spawn()

  // The image, built once. The server answers a tag in its full form, as
  // `docker.io/library/ailoy-cad:1`.
  if (!(await client.listImage()).some((image) => image.reference.split('/').pop() === IMAGE)) {
    console.log(`Building ${IMAGE}, once; it takes a few minutes.`)
    const recipe = new Recipe('python:3.12-slim-trixie')
      // OpenCascade's wheel links libGL and libX11, which the slim image leaves out.
      .run(
        'apt-get update && apt-get install -y --no-install-recommends ' +
          'libgl1 libx11-6 && rm -rf /var/lib/apt/lists/*',
      )
      .run('pip install --no-cache-dir vtk==9.6.2')
      .run('pip install --no-cache-dir cadquery-ocp==7.9.3.1.1')
      .run(
        'pip install --no-cache-dir --no-deps cadquery==2.8.0 ' +
          '&& pip install --no-cache-dir casadi ezdxf multimethod nlopt pyparsing ' +
          'runtype scipy typing_extensions trimesh numpy pillow',
      )
    for await (const entry of client.buildImage(recipe, IMAGE)) {
      if ('end' in entry && entry.end.error) throw new Error(`building ${IMAGE}: ${entry.end.error}`)
    }
  }

  // The volumes, made the first time and kept on the server: `409` says one is there already.
  const volumes = []
  for (const name of ['ailoy-cad-skill', 'ailoy-cad-context', 'ailoy-cad-artifacts']) {
    volumes.push(await client.createVolume(name).catch((e) => {
      if (e.status !== 409) throw e
      return client.getVolume(name)
    }))
  }
  const [skill, context, artifacts] = volumes

  // The skill onto the first.
  for (const name of ['SKILL.md', 'render.py']) {
    await skill.writeFile(name, readFileSync(join(SHARED, name)))
  }

  // `context/`, every file of it, onto the second.
  for (const to of readdirSync(join(PROJECT, 'context'), { recursive: true }).sort()) {
    const path = join(PROJECT, 'context', to)
    if (statSync(path).isDirectory()) await context.createDirAll(to)
    else await context.writeFile(to, readFileSync(path))
  }

  let agent
  try {
    const sandbox = await client.createSandbox(IMAGE, {
      volumes: [
        { volume: skill, path: '/skills/cad', readonly: true },
        { volume: context, path: '/context', readonly: true },
        { volume: artifacts, path: '/artifacts' },
      ],
      vcpus: 4,
      memoryMib: 4096,
    })

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'openai/gpt-6-astra')
      .instruction(INSTRUCTION)
      .systemTools()
      .sandbox(sandbox)
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
await main(process.argv.slice(2).join(' ') || QUERY)
