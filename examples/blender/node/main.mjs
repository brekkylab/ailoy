// Build 3D scenes in a live Blender that you watch over VNC, with an agent that drives it,
// renders what it built, looks at it and fixes it.
//
//     node main.mjs
//     node main.mjs "A low-poly fox sitting on a mossy rock, for a game"
//
// The Node side of the Rust `blender` example, whose `main.rs` has the long form. Blender runs
// on a VNC display in a console of its own, published here at `localhost:5901` (password
// `ailoy`): open `vnc://localhost:5901` to watch. The agent works in a second console and sends
// its scripts to Blender with `run.py`, at port 8765. The skill (`SKILL.md`, `server.py`,
// `run.py`, `start.sh`, mounted from memory at `/skills/blender` in both consoles) is the one
// the three sides share, in `examples/blender/shared`. What a run uses is in this folder:
//
// * `context/` at `/context`, read-only — what the request is about, when it is not in the
//   prompt: a reference picture, a concept sketch, a `.blend` to start from.
// * `artifacts/` at `/artifacts`, writable — the scripts, `scene.blend` and `scene.glb`, the
//   renders and the turntable.
//
// Environment, also read from `.env`:
//
// * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to be
//   set up as `docs/guide/model-providers.md` says. It has to take images, or it cannot see what it
//   built.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'
import virtx from '@brekkylab/virtx'

const HERE = dirname(fileURLToPath(import.meta.url))
// Absolute, because a mount is named to the server as a `file://` URL.
// What the Rust, Python and Node sides share: the skill.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE

const { AgentBuilder } = ailoy
const { ConsoleClient, Directory, HostMount, Recipe, ensureVirtx } = virtx

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
  'Build a small isometric diorama of a ramen stall at night: a square base about 4 m across, ' +
  'the stall with a counter, three stools and a noren curtain over the front, paper lanterns ' +
  'that glow, a steaming bowl on the counter and a vending machine beside it. Warm light from ' +
  'inside the stall, cool moonlight from above. Frame it from the front three-quarters, render ' +
  'it, and give me a turntable.'

const SKILL_FILES = ['SKILL.md', 'server.py', 'run.py', 'start.sh']

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
    SKILL_FILES.reduce(
      (dir, name) => dir.withFile(name, readFileSync(join(SHARED, name))),
      new Directory(),
    ),
    join(PROJECT, 'skill'),
  )

  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()

  let blender, consoleClient, agent
  try {
    // The agent's console is stopped between its tool calls, and Blender with it would be, so
    // Blender has a console the agent does not hold.
    blender = await ConsoleClient.builder()
      .image(
        new Recipe('debian:sid-slim')
          // Blender 5.2 is in sid for every architecture. `mesa-vulkan-drivers` carries the
          // guest's venus ICD and llvmpipe, `libvulkan1` the loader; Xvnc is the display.
          .step(
            'apt-get update && apt-get install -y --no-install-recommends ' +
              'blender python3-numpy mesa-vulkan-drivers libvulkan1 ' +
              'tigervnc-standalone-server tigervnc-tools && rm -rf /var/lib/apt/lists/*',
          )
          // Saved preferences skip the first-run splash, and renders open no window.
          .step(
            'blender -b --factory-startup --python-expr "import bpy; ' +
              'v = bpy.context.preferences.view; v.show_splash = False; ' +
              "v.render_display_type = 'NONE'; bpy.ops.wm.save_userpref()\"",
          ),
      )
      .mountReadonly(skill, '/skills/blender')
      .mountReadonly(join(PROJECT, 'context'), '/context')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      // What comes in is the viewer, at 5901 here to leave 5900 to macOS Screen Sharing, and
      // `run.py` from the agent's console.
      .network(true)
      .ports(['5901:5900', '8765:8765'])
      .gpu(true)
      .vcpus(4)
      .memoryMib(8192)
      .gpuMemoryMib(8192)
      .build()

    // Blender up before the agent's first turn, so the person can watch from the start.
    const started = await blender.exec(['sh', '/skills/blender/start.sh'], 180_000)
    if (started.code !== 0) throw new Error(`starting Blender: ${started.stderr.toString()}`)
    console.log('Blender is up: open vnc://localhost:5901 (password: ailoy) to watch.\n')

    // Python for `run.py`, and a network to reach the port Blender's console published.
    consoleClient = await ConsoleClient.builder()
      .image(new Recipe('python:3.12-slim-trixie'))
      .mountReadonly(skill, '/skills/blender')
      .mountReadonly(join(PROJECT, 'context'), '/context')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      .network(true)
      .build()
    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'openai/gpt-6-astra')
      .instruction(INSTRUCTION)
      .systemTools()
      .console(consoleClient)
      .skill('/skills/blender')
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

    // Blender's console, and Blender in it, goes when this returns.
    const stdin = createInterface({
      input: process.stdin,
      output: process.stdout,
    })
    await stdin.question('\nBlender stays open at vnc://localhost:5901. Press Enter to close it.')
    stdin.close()
  } finally {
    await agent?.close()
    await consoleClient?.close()
    await blender?.close()
    await skill.unmount()
  }
}

loadDotenv()
await main(process.argv.slice(2).join(' ') || QUERY)
