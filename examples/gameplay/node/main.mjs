// An agent that plays OpenTTD, running in a sandbox, while you watch it over VNC.
//
//     node main.mjs
//     node main.mjs "Connect every town over 1000 people by air"
//
// The Node side of the Rust `gameplay` example, whose `main.rs` has the long form. The game
// runs as a dedicated server in a sandbox, and the agent runs a company in it from the same
// sandbox, through the skill's `ttd.py`. `examples/gameplay/shared`, the skill (`SKILL.md`,
// `ttd.py`, `admin.py`) and the game's side (`server/`: `start.sh`, the Game Script, the AI and
// `shot.py`) that the three sides share, goes each run onto a volume of the virtx server
// mounted at `/skills/openttd`.
//
// * The volume `ailoy-gameplay-artifacts` at `/artifacts`, writable — the agent's notes, the
//   screenshots it took, the saved games in `user/openttd/save/` (the last as
//   `ailoy-final.sav`), and the logs. It stays on the server between runs, and is copied to
//   `artifacts/` here at the end.
//
// Watch at the `vnc://` address it prints (Screen Sharing on macOS, or any VNC viewer) with
// the password `openttd`.
//
// Environment, also read from `.env`:
//
// * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//   has to be set up as `docs/src/content/docs/guide/model-providers.mdx` says. It reads
//   screenshots when it looks, so it should take images.
// * `OPENTTD_YEARS` — how many years of the game the agent plays, `3` by default. A game month
//   takes a minute to pass.
// * `OPENTTD_SAVE` — a saved game in `user/openttd/save/` of the artifacts to go on with, such
//   as `ailoy-final.sav`, rather than a new game.
// * `OPENTTD_SEED`, `OPENTTD_YEAR`, `OPENTTD_MAP_X`, `OPENTTD_MAP_Y` — the new game's random
//   seed, its first year (`1950`), and its size as powers of two (`8`, 256 tiles).
// * `OPENTTD_SIZE` — the viewer's display, `1280x720` by default.
// * `OPENTTD_VNC_PASSWORD` — the viewer's password, `openttd` by default.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'
import virtx from '@brekkylab/virtx'

const HERE = dirname(fileURLToPath(import.meta.url))
// What the Rust, Python and Node sides share: the skill and the game's side.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE

const { AgentBuilder } = ailoy
const { Client, PortForwarder, Recipe } = virtx

// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE = 'ailoy-gameplay:1'

// The VNC server's port in the sandbox.
const VNC_PORT = 5900

// The goal when none is given.
const GOAL =
  'Make Ailoy Transport as valuable as you can: build routes that earn, grow the ones that ' +
  'work, and have the loan paid back by the end if you can.'

// Rounds at most, so that an agent that answers at once each time does not go on forever.
const MAX_ROUNDS = 200

// The settings `start.sh` reads, passed on to it when they are set here.
const GAME_SETTINGS = [
  'OPENTTD_SEED',
  'OPENTTD_YEAR',
  'OPENTTD_MAP_X',
  'OPENTTD_MAP_Y',
  'OPENTTD_ADMIN_PASSWORD',
]

const INSTRUCTION =
  'You play OpenTTD: a game is running in your sandbox, and one company in it is yours. ' +
  'Read the openttd skill before anything else, and play with its `ttd.py`.\n\n' +
  'You play in rounds. In each, look at how the company is doing, decide what to build or ' +
  'change, do it, and let time pass with `wait`, a month or two at a time, checking the ' +
  'report each time. End a round with a few lines on what you did and how it is going; the ' +
  'next begins with the date.\n\n' +
  '# Artifacts\n\n' +
  'Path: `/artifacts`\n\n' +
  'Keep your notes in `/artifacts/notes.md`: the routes, their stations, depots and ' +
  'vehicles, what each earns, and your plan. Update them as you go. Command output from ' +
  'earlier rounds drops out of what you remember; the notes do not.'

// From the nearest `.env` up from this file. What the environment already has wins.
function loadDotenv() {
  for (let dir = HERE; ; dir = dirname(dir)) {
    const path = join(dir, '.env')
    if (existsSync(path)) return process.loadEnvFile(path)
    if (dirname(dir) === dir) return
  }
}

async function main(goal) {
  const years = Number(process.env.OPENTTD_YEARS ?? 3)
  if (!Number.isInteger(years)) throw new Error('OPENTTD_YEARS is a number of years')
  const size = process.env.OPENTTD_SIZE ?? '1280x720'
  const password = process.env.OPENTTD_VNC_PASSWORD ?? 'openttd'
  const save = process.env.OPENTTD_SAVE ?? ''

  // The virtx server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none. The sandboxes it holds go when it is closed.
  const client = Client.spawn()

  // The image, built once: the game, and the free graphics, sounds and music it needs, a
  // display for the viewer, Python for `ttd.py`, and ImageMagick for the agent's screenshots.
  // The server answers a tag in its full form, as `docker.io/library/ailoy-gameplay:1`.
  if (!(await client.listImage()).some((image) => image.reference.split('/').pop() === IMAGE)) {
    console.log(`Building ${IMAGE}, once; it takes a few minutes.`)
    const recipe = new Recipe('debian:trixie-slim').run(
      'apt-get update ' +
        '&& apt-get install -y --no-install-recommends ' +
        'openttd openttd-opengfx openttd-opensfx openttd-openmsx xvfb x11vnc python3 ' +
        'imagemagick ' +
        '&& rm -rf /var/lib/apt/lists/*',
    )
    for await (const entry of client.buildImage(recipe, IMAGE)) {
      if ('end' in entry && entry.end.error) throw new Error(`building ${IMAGE}: ${entry.end.error}`)
    }
  }

  // The volumes, made the first time and kept on the server: `409` says one is there already.
  const volumes = []
  for (const name of ['ailoy-gameplay', 'ailoy-gameplay-artifacts']) {
    volumes.push(await client.createVolume(name).catch((e) => {
      if (e.status !== 409) throw e
      return client.getVolume(name)
    }))
  }
  const [files, artifacts] = volumes

  // `shared/`, every file of it, onto the first.
  for (const to of readdirSync(SHARED, { recursive: true }).sort()) {
    const path = join(SHARED, to)
    if (statSync(path).isDirectory()) await files.createDirAll(to)
    else await files.writeFile(to, readFileSync(path))
  }

  const sandbox = await client.createSandbox(IMAGE, {
    volumes: [
      { volume: files, path: '/skills/openttd', readonly: true },
      { volume: artifacts, path: '/artifacts' },
    ],
    ports: [VNC_PORT],
    vcpus: 2,
    memoryMib: 2048,
  })

  // An exec takes no environment of its own, so the settings go through `env`.
  const start = ['env']
  for (const name of GAME_SETTINGS) {
    if (process.env[name] !== undefined) start.push(`${name}=${process.env[name]}`)
  }
  start.push('sh', '/skills/openttd/server/start.sh', size, password, save)
  // The game is one job, running for as long as the game does; `start.sh` says `ready` once it
  // is up. Leaving its log leaves the job running.
  for await (const entry of sandbox.exec(start)) {
    if (entry.stdout === 'ready') break
    if ('stdout' in entry) console.log(entry.stdout)
    else if ('stderr' in entry) console.log(entry.stderr)
    else if ('end' in entry) throw new Error(`OpenTTD did not start (exit ${entry.end.code})`)
  }
  const viewer = PortForwarder.open(client, sandbox.ports[0].url)
  console.log(`OpenTTD is running. Watch it at vnc://${viewer.address} (password: ${password}).`)

  let date = await gameDate(sandbox)
  const until = yearOf(date) + years

  const agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5')
    .instruction(INSTRUCTION)
    .sandbox(sandbox)
    .skill('/skills/openttd')
    // Many commands a round, each with its output: keep the last two rounds' results
    // whole, and of the screenshots the agent reads, the latest ten once more than
    // thirty have built up.
    .contextManager({ maxInputTokens: 80_000, preserveRecentTurns: 2 })
    .build()

  let prompt =
    `It is ${date}, and the game is yours until ${until}-01-01. ${goal}\n\n` +
    'Start by reading the skill, then look at the company, the largest towns and the ' +
    'industries, and build a first route that will earn.'
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    console.log(`\n=== Round ${round}: ${date} ===\n`)
    await play(agent, prompt)

    date = await gameDate(sandbox)
    if (yearOf(date) >= until) {
      console.log(`\n=== The game has reached ${date} ===`)
      break
    }
    prompt =
      `It is ${date}; the game is yours until ${until}-01-01. Go on: read your notes, check ` +
      'the report, then fix what needs fixing, grow what earns, and let time pass.'
  }

  const saving = sandbox.exec(['python3', '/skills/openttd/ttd.py', 'save', 'ailoy-final'], { timeoutMs: 60_000 })
  for await (const entry of saving) if ('stdout' in entry) console.log(entry.stdout)

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
  const rl = createInterface({ input: process.stdin, output: process.stdout })
  await rl.question('The game is still there to look at. Press Enter to stop.\n')
  rl.close()

  await agent.close()
  viewer.close()
  await client.close()
}

// Run one round of the agent, printing what it says and does.
async function play(agent, prompt) {
  for await (const { message, finish_reason } of agent.run(prompt)) {
    // The run ends on any other reason as well, and without this it ends in silence.
    if (finish_reason.type === 'length') {
      console.error('(the reply was cut off at the token limit)')
    }
    if (message.role === 'assistant') {
      for (const part of message.contents) {
        if (part.type === 'text') console.log(part.text)
      }
      for (const call of message.tool_calls ?? []) {
        console.log(`→ ${call.function.name} ${JSON.stringify(call.function.arguments)}`)
      }
    }
    // A screenshot is an image part: its bytes are no use on a terminal. And the rest only as
    // much as shows what came back.
    else if (message.role === 'tool') {
      for (const part of message.contents) {
        if (part.type === 'image') {
          console.log('← [image]')
        } else {
          const text = JSON.stringify(part)
          console.log(`← ${text.slice(0, 600)}${text.length > 600 ? ' …' : ''}`)
        }
      }
    }
  }
}

// The game's date, as `YYYY-MM-DD`.
async function gameDate(sandbox) {
  let stdout = ''
  for await (const entry of sandbox.exec(['python3', '/skills/openttd/ttd.py', '--json', 'status'], { timeoutMs: 60_000 })) {
    if ('stdout' in entry) stdout += entry.stdout
    else if ('stderr' in entry) console.error(entry.stderr)
    else if ('end' in entry && entry.end.code !== 0) throw new Error('asking the game its date failed')
  }
  const { date } = JSON.parse(stdout)
  if (typeof date !== 'string') throw new Error("the game's status has no date")
  return date
}

function yearOf(date) {
  const year = Number(date.split('-')[0])
  if (!Number.isInteger(year)) throw new Error(`${date} is not a date`)
  return year
}

loadDotenv()
await main(process.argv.slice(2).join(' ') || GOAL)
