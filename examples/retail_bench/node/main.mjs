// Run RetailBench (https://github.com/linghuazhang01/RetailBench) against an agent: one
// supermarket, one day at a time, for as many days as the store survives.
//
//     node main.mjs --smoke      # eight SKUs, two days: the machinery
//     node main.mjs --days 3     # three days of the real store
//     node main.mjs              # the benchmark: 96 SKUs, 180 days
//
// The Node side of the Rust `retail_bench` example, whose `main.rs` has the long form: how a
// day is a turn, why three of RetailBench's tools stay tools and the rest are files, and what a
// run leaves behind. The prompts (`system.md`, `user.md`, `nudge.md`) and the store's simulator
// (`simulator/sim`, which fetches RetailBench's code and data on the first run) are the ones
// the three sides share, in `examples/retail_bench/shared`. Each run writes its record to
// `runs/` in this folder:
//
//     runs/<slug>/
//       metrics.json      run_days, final_networth, total_sales, and the ratios
//       days.jsonl        one line per closed day: funds, net worth, sales
//       tool_calls.jsonl  one line per action, in RetailBench's own field names
//       days/NNN.json     the messages of one day's turn
//       context/          the tree as it stood when the run ended, at `/context`, read-only
//       artifacts/        what the agent wrote, notes/<date>.md among it, at `/artifacts`
//
// The store stops with the run; `--keep-store` leaves it up to be asked
// (`simulator/sim view view_inventory --run runs/<slug>`).
//
// Environment, also read from `.env`:
//
// * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's API
//   key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
//
// `npm install` here first, after building the addon (`npm run build` in `bindings/node`).

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import ailoy from '@brekkylab/ailoy'

const HERE = dirname(fileURLToPath(import.meta.url))
// Absolute, because a mount is named to the console server as a `file://` URL.
// What the Rust, Python and Node sides share: the prompts and the store's
// simulator, which keeps the upstream code and the dataset it fetched beside itself.
const SHARED = resolve(HERE, '../shared')
// What a run reads and writes, beside this file.
const PROJECT = HERE
const SIM = join(SHARED, 'simulator', 'sim')

const {
  AgentBuilder,
  ConsoleClient,
  Recipe,
  addAgentProvider,
  addToolProvider,
  registerTool,
  ensureVirtx,
} = ailoy

const SYSTEM = readFileSync(join(SHARED, 'system.md'), 'utf8')
// Sent every morning.
const USER = readFileSync(join(SHARED, 'user.md'), 'utf8')
// Sent when a turn ends without closing the day.
const NUDGE = readFileSync(join(SHARED, 'nudge.md'), 'utf8')

// The tools the agent may call, `end_today` last because that is where it belongs in a day.
const ACTIONS = ['place_order', 'modify_sku_price', 'end_today']

// The name the tool and agent providers carrying the three actions are registered under.
const PROVIDER = 'retail_bench'

// How many times a day is asked before the harness closes it itself: a day left open would
// stop the clock, and the run would never end.
const NUDGES = 3

const HELP = `\
node main.mjs [flags]

  --days N         the horizon (default: 180)
  --config NAME    dynamic_hard | dynamic_middle | still_hard | still_middle
  --smoke          eight SKUs and two days: the machinery, not a result
  --keep-store     leave the store's server running when the run ends
`

function parseArgs(argv) {
  const args = { days: undefined, config: 'dynamic_hard', smoke: false, keepStore: false }
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]
    if (flag === '--days') {
      args.days = Number(argv[++i])
      if (!Number.isInteger(args.days) || args.days < 0) throw new Error('--days needs a number')
    } else if (flag === '--config') {
      args.config = argv[++i]
      if (args.config === undefined) throw new Error('--config needs a name')
    } else if (flag === '--smoke') {
      args.smoke = true
    } else if (flag === '--keep-store') {
      args.keepStore = true
    } else if (flag === '-h' || flag === '--help') {
      process.stdout.write(HELP)
      process.exit(0)
    } else {
      throw new Error(`unknown flag ${flag}\n\n${HELP}`)
    }
  }
  args.days ??= args.smoke ? 2 : 180
  return args
}

// From the nearest `.env` up from this file. What the environment already has wins.
function loadDotenv() {
  for (let dir = HERE; ; dir = dirname(dir)) {
    const path = join(dir, '.env')
    if (existsSync(path)) return process.loadEnvFile(path)
    if (dirname(dir) === dir) return
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const model = process.env.AILOY_MODEL ?? 'openai/gpt-6-astra'

  await prepare(args.smoke)

  const runDir = join(PROJECT, 'runs', slug(model, args.config))
  const context = join(runDir, 'context')
  const artifacts = join(runDir, 'artifacts')
  for (const dir of [context, join(artifacts, 'notes'), join(runDir, 'days')]) {
    mkdirSync(dir, { recursive: true })
  }
  const record = new Record(runDir)

  // ── the store
  const store = await Store.start(runDir, context, args.config, args.days)

  // ── the agent
  const day = { over: false }
  const tools = await registerActions(store, record, day)
  // The console server, fetched into virtx's cache the first time: a host that installed only
  // ailoy has none.
  await ensureVirtx()
  const consoleClient = await ConsoleClient.builder()
    .image(new Recipe('python:3.12-slim-trixie'))
    .mountReadonly(context, '/context')
    .mount(artifacts, '/artifacts')
    .network(false)
    .vcpus(2)
    .memoryMib(2048)
    .build()
  // One agent a day, each starting from the system message alone, all sharing the console.
  const newAgent = () =>
    new AgentBuilder(model)
      .agentProvider(PROVIDER)
      .instruction(SYSTEM)
      .maxTokens(64000)
      .systemTools()
      .tools(tools)
      .console(consoleClient)
      .build()

  console.log(`  model  ${model}`)
  console.log(`  store  ${store.base}`)
  console.log(`  run    ${runDir}\n`)

  // ── the days
  let failure = null
  try {
    for (let n = 1; n <= args.days; n++) {
      try {
        const reason = await runDay(newAgent, store, record, day, n, args.days)
        if (reason !== null) {
          console.log(`\nthe run ended on day ${n}: ${reason}`)
          break
        }
      } catch (e) {
        // A failed day ends the run and not the process: the days before it are a result, and
        // late in a run that is hours of work.
        console.log(`\nday ${n} failed: ${e.message}`)
        failure = `day ${n}: ${e.message}`
        break
      }
    }

    const metrics = await store.get('/metrics').catch(() => null)
    record.write('metrics.json', {
      model,
      config: args.config,
      horizon: args.days,
      failed: failure,
      metrics,
    })
    console.log(`\n${JSON.stringify(metrics, null, 2)}`)
    console.log(`\nwritten to ${runDir}`)
  } finally {
    await consoleClient.close()
    if (args.keepStore) {
      console.log(`the store is still up at ${store.base} — \`sim stop --run ${runDir}\` ends it`)
    } else {
      await store.post('/stop', {}).catch(() => {})
    }
  }
}

// One day: write its tree, ask the agent until it closes the day, and write down what the day
// came to. Returns why the run ended, if this day ended it, and `null` otherwise.
async function runDay(newAgent, store, record, day, n, maxDays) {
  // Written before the turn, so the agent's first read is of a tree whose date matches the
  // prompt. Day one's was written when the store booted.
  if (n > 1) await store.post('/context', { market: true })
  const state = await store.get('/state')
  const fill = (template) =>
    template
      .replaceAll('{{day}}', String(n))
      .replaceAll('{{max_days}}', String(maxDays))
      .replaceAll('{{date}}', typeof state.date === 'string' ? state.date : '')
      .replaceAll('{{funds}}', number(state.funds))
      .replaceAll('{{net_worth}}', number(state.net_worth))

  // An empty history every morning: the system message, and nothing the agent did yesterday
  // except what it wrote into its notes.
  const agent = await newAgent()
  day.over = false

  // Whether the model answered at all today. A day where it never did is a model that could
  // not be reached, not one that decided to do nothing, and is not played as an empty day.
  let heard = false
  let error = null
  try {
    for (let nudge = 0; nudge < NUDGES && !day.over; nudge++) {
      try {
        for await (const { message } of agent.run(fill(nudge === 0 ? USER : NUDGE))) {
          heard = true
          show(n, message)
          // The day closes inside a tool call, so this is checked after every message: what
          // the model would say next belongs to a day that has not been written yet.
          if (day.over) break
        }
      } catch (e) {
        error = e.message
      }
    }
    record.write(`days/${String(n).padStart(3, '0')}.json`, {
      day: n,
      state,
      error,
      messages: agent.history,
    })
  } finally {
    await agent.close()
  }

  let ended = null
  if (!day.over) {
    if (!heard) throw new Error(`the model never answered: ${error ?? ''}`)
    console.log(`${pad(n, 3)}   closed by the harness: the agent did not call end_today`)
    const call = await store.act('end_today', {})
    await record.toolCall('end_today', { forced: true }, call, store, 0)
    ended = call.terminated
  }

  const metrics = await store.get('/metrics')
  record.append('days.jsonl', {
    day: n,
    date: state.date ?? null,
    funds: metrics.final_funds ?? null,
    net_worth: metrics.final_networth ?? null,
    total_sales: metrics.total_sales ?? null,
    refusals: metrics.refusals ?? null,
  })
  console.log(
    `day ${pad(n, 3)}  funds ${pad(number(metrics.final_funds), 12)}` +
      `  net worth ${pad(number(metrics.final_networth), 12)}` +
      `  sales ${pad(number(metrics.total_sales), 8)}`,
  )
  if (ended) return ended
  return typeof metrics.terminated === 'string' ? metrics.terminated : null
}

// Register the three actions under `PROVIDER` and return their descriptions, whose schemas are
// the store's own rather than written here.
//
// A tool description in a spec is resolved by name against a tool provider, so the actions go
// into one of their own, which starts with every built-in tool as the default one does.
async function registerActions(store, record, day) {
  const { actions: declared = [] } = await store.get('/actions')

  addToolProvider(PROVIDER)
  const descs = []
  for (const name of ACTIONS) {
    // A name the store does not declare means this example and the simulator have drifted
    // apart, which is worth stopping for rather than running without the tool.
    const spec = declared.find((spec) => spec.name === name)
    if (!spec) throw new Error(`the store does not declare '${name}'`)
    let description = spec.description ?? ''
    if (name === 'end_today') {
      // Holds only in this harness, so the store's own description does not say it.
      description +=
        ' This ends your turn: call it once, when you are done for the day, and expect ' +
        'no further instructions afterwards.'
    }
    const desc = { name, description, parameters: spec.input_schema }
    const func = async (args) => {
      const started = performance.now()
      try {
        const call = await store.act(name, args)
        const elapsed = (performance.now() - started) / 1000
        await record.toolCall(name, args, call, store, elapsed)
        if (call.dayOver) day.over = true
        return said(call)
      } catch (e) {
        // Nothing the model can fix, but saying so beats a silent failure: the day never
        // closes, and the harness closes it.
        return `The store could not be reached: ${e.message}`
      }
    }
    descs.push(registerTool(desc, func, { provider: PROVIDER }))
  }
  addAgentProvider(PROVIDER, { langModelProvider: 'default', toolProvider: PROVIDER })
  return descs
}

// What the model is shown for one action: the store's own words, and a line saying so when the
// day just closed. Without it, a model that was just told yesterday's sales tends to keep going.
function said(call) {
  if (!call.ok) return `Refused: ${call.said}`
  let text = call.said
  if (call.dayOver) {
    text += '\n\n---\nThe day is over. Stop here — do not call anything else.'
    text += call.terminated
      ? ` The run has ended: ${call.terminated}.`
      : " Tomorrow's files will be waiting when you are asked again."
  }
  return text
}

// Run `sim` with `args`, settling with its exit code. Windows can't execute `sim`'s shebang, so
// there it runs under `uv`'s Python.
async function sim(args, { quiet = false } = {}) {
  const [command, prefix] =
    process.platform === 'win32'
      ? [process.env.UV ?? 'uv', ['run', '--no-project', 'python', SIM]]
      : [SIM, []]
  const child = spawn(command, [...prefix, ...args], { stdio: quiet ? 'ignore' : 'inherit' })
  const [code, signal] = await once(child, 'exit')
  return code ?? signal
}

// Set the store up if it is not: the upstream code, an interpreter for it, and the dataset.
async function prepare(smoke) {
  const ready = !smoke && (await sim(['status'], { quiet: true }).catch(() => null)) === 0
  if (ready) return
  const status = await sim(['setup', '--no-serve', ...(smoke ? ['--skus', '8'] : [])]).catch(
    (e) => {
      throw new Error(
        process.platform === 'win32'
          ? `running ${SIM} through \`uv\`. It needs \`uv\` on PATH, or \`UV\` naming it.`
          : `running ${SIM}. It needs \`python3\` on PATH.`,
        { cause: e },
      )
    },
  )
  if (status !== 0) throw new Error(`setting the store up: ${status}`)
}

// The store, over HTTP.
class Store {
  constructor(base) {
    this.base = base
  }

  // Start this run's server — booting loads the store's data, and `sim serve` waits for it.
  static async start(run, context, config, days) {
    const status = await sim([
      'serve',
      '--run',
      run,
      '--context',
      context,
      '--config',
      config,
      '--days',
      String(days),
    ])
    if (status !== 0) throw new Error(`\`sim serve\` failed: ${status}`)
    const port = readFileSync(join(run, 'sim.port'), 'utf8').trim()
    const store = new Store(`http://127.0.0.1:${port}`)
    await store.get('/state').catch((e) => {
      throw new Error('the store did not answer', { cause: e })
    })
    return store
  }

  async get(path) {
    const response = await fetch(`${this.base}${path}`)
    const body = await response.json()
    if (!response.ok) {
      throw new Error(`GET ${path} failed (${response.status}): ${JSON.stringify(body)}`)
    }
    return body
  }

  async post(path, payload) {
    const response = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const body = await response.json()
    if (!response.ok) {
      throw new Error(`POST ${path} failed (${response.status}): ${JSON.stringify(body)}`)
    }
    return body
  }

  // One action. `409` is the store refusing a well-formed request, which the agent is meant to
  // read and try differently; anything else that is not `200` is this side being wrong. A
  // refusal is an answer, not an error.
  async act(name, args) {
    const response = await fetch(`${this.base}/actions/${name}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(args),
    })
    const body = await response.json()
    if (response.status !== 200 && response.status !== 409) {
      throw new Error(`${name} failed (${response.status}): ${JSON.stringify(body.error ?? null)}`)
    }
    const ok = response.status === 200
    const said = body[ok ? 'formatted' : 'error']
    return {
      said: typeof said === 'string' ? said : '',
      ok,
      dayOver: body.day_over === true,
      terminated: typeof body.terminated === 'string' ? body.terminated : null,
      body,
    }
  }
}

// The run's directory, written to as the run goes, so a run stopped on day 90 has ninety days
// on disk.
class Record {
  constructor(dir) {
    this.dir = dir
  }

  write(name, value) {
    writeFileSync(join(this.dir, name), JSON.stringify(value, null, 2))
  }

  append(name, value) {
    const path = join(this.dir, name)
    try {
      appendFileSync(path, `${JSON.stringify(value)}\n`)
    } catch (e) {
      console.error(`could not append to ${path}: ${e.message}`)
    }
  }

  // One action, in the field names RetailBench's own `analysis/` scripts read.
  async toolCall(tool, args, call, store, elapsed) {
    const state = await store.get('/state').catch(() => ({}))
    this.append('tool_calls.jsonl', {
      ts: unixNow(),
      tool,
      args,
      ok: call.ok,
      result: call.body.result ?? null,
      formatted: call.said,
      funds: state.funds ?? null,
      net_worth: state.net_worth ?? null,
      current_date: state.current_date ?? null,
      day: state.day ?? null,
      elapsed_time: Math.round(elapsed * 10_000) / 10_000,
    })
  }
}

// One line per thing the agent says or calls, so a long run can be watched.
function show(n, message) {
  for (const part of message.contents) {
    if (part.type !== 'text') continue
    const line = part.text.trim().split('\n')[0]
    if (line) console.log(`${pad(n, 3)}   ${truncate(line, 140)}`)
  }
  for (const call of message.tool_calls ?? []) {
    const args = JSON.stringify(call.function.arguments)
    console.log(`${pad(n, 3)} → ${call.function.name}(${truncate(args, 120)})`)
  }
}

function truncate(text, at) {
  const chars = Array.from(text)
  return chars.length <= at ? text : chars.slice(0, at).join('') + '…'
}

const pad = (value, width) => String(value).padStart(width)

const number = (value) => (typeof value === 'number' ? value.toFixed(2) : '-')

const unixNow = () => Math.floor(Date.now() / 1000)

// A sortable name for this run: when it started, which model, which configuration.
const slug = (model, config) => `${unixNow()}_${model.replace(/[/: ]/g, '-')}_${config}`

loadDotenv()
await main()
