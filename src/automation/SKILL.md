---
name: ailoy-automation-definition
description: How to write an ailoy automation — the console.json / workflow.json / tasks/ directory that a run is made from, and the trigger.json + trigger.py that register it with the daemon. Use when creating or fixing an automation definition, a python or agent task, or a trigger script.
---

# Writing an ailoy automation

An automation is a directory. A run is made from it when a trigger fires.

```
<automation>/
  console.json      where tasks run: image, whether there is a network   → console.schema.json
  workflow.json     what runs: tasks, their inputs, the final output      → workflow.schema.json
  tasks/<name>.py   a python task's code   (when workflow.json has no `code`)
  tasks/<name>.md   an agent task's prompt (when workflow.json has no `prompt`)
  trigger.json      draft of the daemon registration: event types, sources    → trigger.schema.json
  trigger.py        the trigger script: decides which events become runs
```

The first four files are the automation and their canonical copy is this directory.
`trigger.json` and `trigger.py` are for the daemon: the JSON is a draft the CLI uploads
(`ailoy-daemon triggers register <dir>`), the script is read by the daemon each time
it runs. Nothing runs until the trigger is registered.

Validate before registering: the daemon rejects a registration whose automation does
not load, and the errors below are what it says.

## console.json

```json
{ "image": "python:3.12-slim", "network": false }
```

- `image`: OCI reference. Must have `python3`; must have `pip` if any task lists packages.
- `network`: `false` (default) or `true`. The server enforces it, so a task that needs
  no network should not ask for one.

## workflow.json

```json
{
  "timeout_secs": 600,
  "output": "report",
  "tasks": [
    { "name": "fetch",  "kind": "python", "inputs": { "event": "event" }, "packages": ["requests==2.32.3"] },
    { "name": "report", "kind": "agent",  "inputs": { "data": "fetch" },
      "agent": { "model": "anthropic/claude-sonnet-5", "instruction": "..." },
      "tools": ["read", "write"],
      "output_schema": { "type": "object", "properties": { "summary": { "type": "string" } }, "required": ["summary"] } }
  ]
}
```

Rules the schema cannot say, and the exact messages a violation produces:

| rule | message |
|---|---|
| task names are unique | `<name>: duplicate task name` |
| `event` is not a task name | `event: \`event\` is reserved for the trigger payload` |
| every `inputs` value is an earlier task name or `event` | `<name>: input \`<input>\` refers to unknown task \`<x>\`` |
| no task reads itself, no cycles | `<name>: input \`<input>\` refers to the task itself`, `cycle among tasks: a, b` |
| `output` names a task | `output: unknown task \`<x>\`` |
| python: `code` inline or `tasks/<name>.py` exists | `<name>: no inline \`code\` and tasks/<name>.py cannot be read` |
| agent: `prompt` inline or `tasks/<name>.md` exists | `<name>: no inline \`prompt\` and tasks/<name>.md cannot be read` |
| packages are `name==version`, one version per name | `<name>: package \`x\` must be pinned as name==version`, `package \`x\` is pinned to both 1.0 and 2.0` |
| `agent.tools` is empty, `agent.subagents` absent | `<name>: \`agent.tools\` must be empty; list tool names in the task's \`tools\`` |
| `tools` are built-in names | `<name>: unknown tool \`x\`; built-in tools are shell, read, write, edit, apply_patch, imgread, web_fetch, web_search` |
| `output_schema` is valid JSON Schema | `<name>: output_schema: ...` |

Execution: topological order of `inputs`, ties broken by list order, one task at a
time, all on one console. A task that fails or times out ends the run; the tasks after
it are recorded as `skipped`.

### What flows between tasks

Every task's output is one JSON value. A task's input is one JSON object whose keys
are its `inputs` names: `{ "data": <fetch's output>, "event": <trigger payload> }`.
Keep outputs small and shaped for the reader; a big file goes into the scratch
directory (the working directory) and its path goes into the output.

### python tasks

```
python3 main.py <input.json> <output.json>
```

- Read `sys.argv[1]` (the input object), write the output JSON to `sys.argv[2]`.
  No output file means the output is `null`. A non-zero exit fails the task; stderr is
  kept in the record.
- The working directory is the run's scratch directory, shared by every task of the
  run and kept across retries. Relative paths land there. Files for a person go to
  the artifacts directory (`/artifacts` on the micro-VM console).
- No environment variables are passed. Use only the standard library plus the pinned
  `packages`; they are baked into the image before the run.

### agent tasks

- The agent receives **only the rendered prompt**. Nothing else is attached: no raw
  event, no input file. Put into the template exactly what the agent should see, and
  make the earlier tasks produce values that render well.
- Template syntax: `{{ name }}`, `{{ name.key }}`, `{{ name.list[0].key }}`, looked up
  in the input object. Strings render as themselves, anything else as JSON. A path
  that resolves to nothing stays as written, so a leftover `{{ ... }}` in a run's
  record means a typo.
- The agent gets the console's scratch and artifacts trees (their paths are told to
  it) and the tools listed under the task's `tools`. It does not get tools from
  `agent.tools` and may not have `subagents`.
- Output: the text of the final message. With `output_schema`, the final message must
  be that JSON and the task's output is the parsed value; otherwise the output is the
  text as a JSON string.
- The runner prepends a fixed instruction (this is an automation step, no one to
  ask, results go to artifacts, end with the result); `agent.instruction` follows it.

## Registering: trigger.json and trigger.py

Events have a **type**, and a type exists as soon as an event of it exists: posted from
outside (`POST /events/{type}`), made by a source a trigger declares for itself (a
clock, a watched directory), or published by the daemon when a run ends
(`runs:<trigger>`). Nothing is registered ahead of time. A **trigger** names the types
that wake it, points at an automation directory, and is asked for the next run
whenever one of those types gets a new event.

```json
{
  "automation": "/abs/path/to/this/directory",
  "sources": {
    "tick":    { "kind": "cron", "schedule": "*/1 * * * *" },
    "arrival": { "kind": "fs", "path": "/srv/inbox", "events": ["create"] }
  },
  "watch": ["tickets", "runs:ingest"],
  "script_timeout_secs": 30
}
```

- `sources`: events this trigger makes for itself; the key is the type. `cron`
  publishes `{ "at" }`, `fs` publishes `{ "path", "event" }`.
- `watch`: types made elsewhere. `runs:<trigger>` carries another trigger's run
  results (`{ trigger, run_id, status, output, error, artifacts_url }`).
- The trigger wakes on, and its script can query, exactly `sources` keys + `watch`.
- Runs of one trigger execute one at a time, oldest first. A run that fails stays
  `failed`; to try again, make a new run with `ailoy-daemon triggers fire <name>`.
- `trigger.py` beside the automation decides the next run when that file exists.
  Without it, the oldest event without a run becomes one, with payload
  `{ "type", "at", "payload" }`.
- `script_timeout_secs` (default 30) bounds one call of `trigger.py`.
- Registering and everything else but publishing go over the daemon's socket file
  (`--socket`, `AILOY_DAEMON_SOCKET`, `daemon.sock` in the current directory). Whoever
  may open that file may administer the daemon; there is no token.
- The daemon's TCP address carries `POST /events/{type}` alone, for senders out on a
  network. Issue each of them a credential for its one type:
  `ailoy-daemon tokens issue tickets` opens `POST /events/tickets` and nothing
  else, and prints the secret once.
- The config is applied when registered; edits to `trigger.json` do nothing until
  `triggers register` runs again. Registering is idempotent: the same directory
  uploaded twice leaves one trigger, so re-register whenever you touch the file rather
  than tracking whether you already did. Edits to `trigger.py` and to the automation
  directory apply from the next call or run.

### trigger.py

Same argv shape as a python task, stateless, read-only SQL over a snapshot. **One
call answers at most one run.** If it answers a run, it is called again right after,
until it answers `null`.

```
python3 trigger.py <input.json> <output.json>
```

```json
// input.json
{ "trigger": "data-check", "now": 1758600005, "db": "/scratch/data-check/snapshot.sqlite" }
// output.json
{ "run": { ...payload of the next run... }, "event_id": 41 }   // event_id optional
{ "run": null }                                                // nothing to do
```

The snapshot holds only this trigger's event types and its own runs:

```sql
events(id, type, at, payload)       -- events of this trigger's types; payload is JSON text
runs(id, event_id, created_at, status, payload)   -- runs this trigger made
```

`runs.event_id` is what the script answered as `event_id`, so "events without a run"
is one join and there is nothing else to remember:

```python
import json, sqlite3, sys
inp = json.load(open(sys.argv[1]))
db = sqlite3.connect(f"file:{inp['db']}?mode=ro", uri=True)

# the next event nobody has run for
row = db.execute("""SELECT e.id, e.payload FROM events e LEFT JOIN runs r ON r.event_id = e.id
                    WHERE e.type = 'tickets' AND r.id IS NULL ORDER BY e.id LIMIT 1""").fetchone()

# already ran for this key? (dedup on a field rather than on the event)
done = {json.loads(p).get("id") for (p,) in db.execute("SELECT payload FROM runs")}

# quiet for 5 minutes? (debounce)
(last,) = db.execute("SELECT MAX(at) FROM events WHERE type = 'edits'").fetchone()
quiet = last is not None and inp["now"] - last >= 300

# a batch: one run for everything since the last run
(since,) = db.execute("SELECT COALESCE(MAX(created_at), 0) FROM runs").fetchone()

out = {"run": None} if row is None else {"run": json.loads(row[1]), "event_id": row[0]}
json.dump(out, open(sys.argv[2], "w"))
```

Polling an API is a cron source plus a script that reads the API itself, with
`network` on and whatever credential it needs in the daemon's environment.

A script that exits non-zero or writes something other than `{ "run": ... }` makes
no run; the message is shown as the trigger's `last_error`, and the script is asked
again on the next event of its types.

## Checklist

1. `console.json`: image with python3; `network` only when a task needs one.
2. `workflow.json`: task names, inputs, `output`; pinned packages; agent `tools` on the task.
3. `tasks/<name>.py` reads argv[1], writes argv[2]; `tasks/<name>.md` names only input fields that exist.
4. `trigger.json`: `sources` and/or `watch`; the type names are yours to choose.
5. `trigger.py`: find the next event without a run (or your own condition), answer `{"run": payload, "event_id": id}` or `{"run": null}`.
6. Register: `ailoy-daemon triggers register <dir>`; fix whatever it rejects.

Every rule above is shown as JSON where it is stated, so there is nothing else to read
before writing a definition.
