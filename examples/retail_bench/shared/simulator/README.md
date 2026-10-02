# The simulator

RetailBench's store, as a local REST server. One supermarket, one day at a time,
for as many days as it survives.

```sh
./simulator/sim setup --skus 8        # code, interpreter, data, and a server
curl -s localhost:$(cat .cache/run/sim.port)/ | less
```

Everything under `upstream/` is somebody else's code, restored by checksum from
a pinned commit. Everything beside it is ours and is the whole of the interface:

```
sim                 the only entry point
server.py           the REST server
env.py              the session: three actions, the clock, the score
context.py          the tree the agent reads, written fresh and cut to today
SOURCE              the repository and commit both halves are pinned to
MANIFEST.sha256     sha256 of every file under upstream/
PROVENANCE.md       what was taken, what was left out, and the licence
upstream/           RetailBench's simulator, unmodified
```

## Why a server

Booting the environment reads about six hundred megabytes — the record seed, the
news pool, the review pool, every supplier's quote history — and takes four
seconds. A run is a few dozen actions a day for up to a hundred and eighty days.
Paying the boot per action would cost three hours a run; paying it once costs
four seconds.

The second reason is that a run can then be *watched*. While an agent works,
another terminal can ask the same store what it looks like:

```sh
curl -s localhost:$PORT/state
curl -s localhost:$PORT/views/view_inventory
```

## The API

Bound to `127.0.0.1` on a port the kernel picks, written to `<run>/sim.port`.
Nothing authenticates anything; it is a local process holding a toy shop.

| | |
| --- | --- |
| `GET /` | what is here, and where the store stands |
| `GET /state` | day, date, funds, net worth, whether it is over |
| `GET /metrics` | `run_days`, `final_networth`, `total_sales`, and the ratios |
| `GET /actions` | the three tools, with the schemas upstream declares |
| `POST /actions/<name>` | take an action — **200** taken, **409** refused |
| `GET /views` | the read-only tools, by name |
| `GET /views/<name>` | what that one says now — `text/plain`, or `?format=json` |
| `GET /context` | where the agent's tree is, and when it was written |
| `POST /context` | write it again — `{"market": false}` for the cheap half |
| `POST /stop` | shut down |

### Three actions, and why only three

RetailBench declares nineteen tools. Fifteen only look, and here what they would
have answered is written into the context directory as files instead. What is
left is what changes the store — and, more to the point, **what can be refused**:

```sh
$ curl -sX POST $PORT/actions/place_order \
    -d '{"supplier_id":"supplier_9","items":[{"sku_id":"8066095605","quantity":10}]}'
{"ok": false, "error": "Unknown supplier: supplier_9"}                    # 409

$ curl -sX POST $PORT/actions/place_order \
    -d '{"supplier_id":"supplier_1","items":[{"sku_id":"8066095605","quantity":999999}]}'
{"ok": false, "error": "Insufficient funds to place the order.
                        Required: 3662646.34, Available: 50000.00"}       # 409
```

A refusal is a `409` and not a `400`: the request was well formed and the store
said no. A caller retries a 409 with different arguments; a 400 means it sent
something that was never going to parse. Which supplier quotes which SKU changes
**by the day**, so this is not a rare path — it is the reason these three stayed
tools while the other fifteen became files.

`add_note` is upstream's fourth state-changing tool and is deliberately not
exposed. The agent's memory here is a file it writes itself, which `context.py`
carries into the next morning; nothing about writing a note can be refused, so
nothing about it needs a tool.

### Everything is recorded

Every action and every refusal lands in `<run>/actions.jsonl`, in order, with
the day it happened on. The refusals are half of what a finished run is worth
reading.

## The commands

| | | |
| --- | --- | --- |
| `sim setup` | all four below, in order | a store that answers |
| `sim restore` | downloads the missing or changed files of the simulator | `upstream/`, `MANIFEST.sha256` |
| `sim python` | builds a venv with numpy, pandas, matplotlib | `.cache/venv/` |
| `sim fetch` | downloads the dataset | `.cache/data/` |
| `sim serve` | starts the server for a run | `<run>/sim.port`, `server.log` |
| `sim status` | reads all of the above and reports | nothing |
| `sim run <action> '<json>'` | one action — exit 0 taken, 1 refused | a line in `actions.jsonl` |
| `sim view <name> [k=v…]` | what one read-only tool says now | nothing |
| `sim stop` | shuts the server down | |

`run` and `view` are thin clients over the same API the harness uses, so the
command in a note is the command that runs:

```sh
./simulator/sim run place_order '{"supplier_id":"supplier_2","items":[{"sku_id":"8066095605","quantity":40}]}'
./simulator/sim run end_today
./simulator/sim view view_inventory
```

## Restoring, and knowing you need to

`MANIFEST.sha256` is the sha256 of every upstream file as it was downloaded.

```sh
./simulator/sim restore --check    # changes nothing; exits non-zero on drift
./simulator/sim restore            # downloads whatever is missing or changed
./simulator/sim restore --force    # downloads everything and rewrites the manifest
```

A *changed* file means somebody edited the vendored tree — upstream cannot change
under you, because `SOURCE` pins a commit and not a branch. That is what the
manifest is for: the simulator is 360 KB of code this benchmark's numbers depend
on, and a local edit to it would otherwise be invisible.

To move to another commit, edit `COMMIT` in [`SOURCE`](SOURCE), then
`sim restore --force` and `sim fetch --force`. Both halves move together: the
environment reads its data by file name and by shape, so a store built from one
commit's code and another's data cannot be compared with anything.

## A smoke test is not a result

`--skus 8` fetches eight SKUs spread across the categories instead of all
ninety-six. It is recorded in `.cache/data/manifest.json`, the environment holds
the run to exactly those SKUs, and `sim status` says so every time. A run on
eight SKUs is for seeing that the machinery works.
