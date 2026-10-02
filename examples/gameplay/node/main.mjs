// An agent that plays OpenTTD, running in a console, while you watch it over VNC.
//
//     node main.mjs
//     node main.mjs "Connect every town over 1000 people by air"
//
// The Node side of the Rust `gameplay` example, whose `main.rs` has the long form. The game
// runs as a dedicated server in a console of its own, and the agent runs a company in it from
// another, through the skill's `ttd.py`. The skill (`SKILL.md`, `ttd.py`, `admin.py`, mounted
// from memory at `/skills/openttd` in both consoles) and the game's side (`server/`: `start.sh`,
// the Game Script, the AI and `shot.py`) are the ones the three sides share, in
// `examples/gameplay/shared`. What a run uses is in this folder:
//
// * `artifacts/` at `/artifacts` in both, writable — the agent's notes, the screenshots it
//   took, the saved games in `user/openttd/save/` (the last as `ailoy-final.sav`), and the
//   logs.
//
// Watch at `vnc://localhost:5901` (Screen Sharing on macOS, or any VNC viewer) with the
// password `openttd`.
//
// Environment, also read from `.env`:
//
// * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...). It reads screenshots
//   when it looks, so it should take images.
// * `OPENTTD_YEARS` — how many years of the game the agent plays, `3` by default. A game month
//   takes a minute to pass.
// * `OPENTTD_SAVE` — a saved game in `artifacts/user/openttd/save/` to go on with, such as
//   `ailoy-final.sav`, rather than a new game.
// * `OPENTTD_SEED`, `OPENTTD_YEAR`, `OPENTTD_MAP_X`, `OPENTTD_MAP_Y` — the new game's random
//   seed, its first year (`1950`), and its size as powers of two (`8`, 256 tiles).
// * `OPENTTD_SIZE` — the viewer's display, `1280x720` by default.
// * `OPENTTD_VNC_PASSWORD` — the viewer's password, `openttd` by default.
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { fileURLToPath } from 'node:url'

import ailoy from 'ailoy-node'

const HERE = dirname(fileURLToPath(import.meta.url))
// Absolute, because a mount is named to the server as a `file://` URL.
// What the Rust, Python and Node sides share: the skill and the game's side.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE

const { AgentBuilder, ConsoleClient, Directory, HostMount, Recipe, ensureVirtx } = ailoy

// The port a VNC viewer connects to here, and the VNC server's in the game's console.
const VIEWER_PORT = 5901
const VNC_PORT = 5900

// The game's admin port and `shot.py`'s, published at the same numbers here so that
// `admin.py` names one address for both consoles.
const ADMIN_PORT = 3977
const SHOT_PORT = 5902

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
  'You play OpenTTD: a game is running in the console, and one company in it is yours. ' +
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

  // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
  for (const name of ['artifacts', 'skill']) {
    mkdirSync(join(PROJECT, name), { recursive: true })
  }

  // One mount, shared: both consoles see the same skill. Held here, not left to garbage
  // collection: Node runs no finalizer on exit, and a mount comes down only when the last
  // holder lets go.
  const skill = new HostMount(
    new Directory()
      .withFile('SKILL.md', readFileSync(join(SHARED, 'SKILL.md')))
      .withFile('ttd.py', readFileSync(join(SHARED, 'ttd.py')))
      .withFile('admin.py', readFileSync(join(SHARED, 'admin.py'))),
    join(PROJECT, 'skill'),
  )

  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()

  let game, consoleClient, agent
  try {
    game = await ConsoleClient.builder()
      .image(
        // All of it in `main`: the game, and the free graphics, sounds and music it needs, a
        // display for the viewer, and ImageMagick for the agent's screenshots of it.
        new Recipe('debian:trixie-slim').step(
          'apt-get update ' +
            '&& apt-get install -y --no-install-recommends ' +
            'openttd openttd-opengfx openttd-opensfx openttd-openmsx xvfb x11vnc python3 ' +
            'imagemagick ' +
            '&& rm -rf /var/lib/apt/lists/*',
        ),
      )
      .mountReadonly(join(SHARED, 'server'), '/example')
      .mountReadonly(skill, '/skills/openttd')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      // The build's `apt-get` runs with the session's network. What comes in is the viewer,
      // and the agent's console at the admin port and `shot.py`.
      .ports([`${VIEWER_PORT}:${VNC_PORT}`, `${ADMIN_PORT}:${ADMIN_PORT}`, `${SHOT_PORT}:${SHOT_PORT}`])
      .vcpus(2)
      .memoryMib(2048)
      .build()

    // An exec takes no environment of its own, so the settings go through `env`.
    const start = ['env']
    for (const name of GAME_SETTINGS) {
      if (process.env[name] !== undefined) start.push(`${name}=${process.env[name]}`)
    }
    start.push('sh', '/example/start.sh', size, password, save)
    const out = await game.exec(start, 180_000)
    process.stdout.write(out.stdout)
    if (out.code !== 0) throw new Error(`starting OpenTTD:\n${out.stderr}`)
    console.log(`OpenTTD is running. Watch it at vnc://localhost:${VIEWER_PORT} (password: ${password}).`)

    let date = await gameDate(game)
    const until = yearOf(date) + years

    // Python for `ttd.py`, and a network to reach the ports the game's console published.
    consoleClient = await ConsoleClient.builder()
      .image(new Recipe('python:3.12-slim-trixie'))
      .mountReadonly(skill, '/skills/openttd')
      .mount(join(PROJECT, 'artifacts'), '/artifacts')
      .network(true)
      .build()

    agent = await new AgentBuilder(process.env.AILOY_MODEL ?? 'anthropic/claude-sonnet-5')
      .instruction(INSTRUCTION)
      .maxTokens(32000)
      .systemTools()
      .console(consoleClient)
      .skill('/skills/openttd')
      // Many commands a round, each with its output: keep the last two rounds whole.
      .contextManager({ maxInputTokens: 80_000, preserveRecentTurns: 2 })
      .build()

    let prompt =
      `It is ${date}, and the game is yours until ${until}-01-01. ${goal}\n\n` +
      'Start by reading the skill, then look at the company, the largest towns and the ' +
      'industries, and build a first route that will earn.'
    for (let round = 1; round <= MAX_ROUNDS; round++) {
      console.log(`\n=== Round ${round}: ${date} ===\n`)
      await play(agent, prompt)

      date = await gameDate(game)
      if (yearOf(date) >= until) {
        console.log(`\n=== The game has reached ${date} ===`)
        break
      }
      prompt =
        `It is ${date}; the game is yours until ${until}-01-01. Go on: read your notes, check ` +
        'the report, then fix what needs fixing, grow what earns, and let time pass.'
    }

    const saved = await game.exec(['python3', '/skills/openttd/ttd.py', 'save', 'ailoy-final'], 60_000)
    process.stdout.write(saved.stdout)
    const rl = createInterface({ input: process.stdin, output: process.stdout })
    await rl.question('The game is still there to look at. Press Enter to stop.\n')
    rl.close()
  } finally {
    // Closing the consoles tears them down, game and all.
    await agent?.close()
    await consoleClient?.close()
    await game?.close()
    await skill.unmount()
  }
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
async function gameDate(consoleClient) {
  const out = await consoleClient.exec(['python3', '/skills/openttd/ttd.py', '--json', 'status'], 60_000)
  if (out.code !== 0) throw new Error(`asking the game its date:\n${out.stderr}`)
  const { date } = JSON.parse(out.stdout)
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
