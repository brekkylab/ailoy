# Where `upstream/` came from

[RetailBench](https://github.com/linghuazhang01/RetailBench)'s simulator, copied
unmodified and restorable by checksum. Nothing in that directory is ours.

| | |
| --- | --- |
| Repository | `linghuazhang01/RetailBench` |
| Commit | `f1d37b35abfa60ef5599e56d2cdbdaa4fd9b8bf3` (2026-07-08) — also in [`SOURCE`](SOURCE) |
| License | MIT — see [`upstream/LICENSE`](upstream/LICENSE) |
| Paper | [arXiv:2603.16453](https://arxiv.org/abs/2603.16453) |

```
upstream/
  retail_environment.py   the environment: state, the nineteen tools, the daily step
  module/                 inventory, orders, suppliers, customers, news, reviews, records
  model/                  the demand, return-rate and review-rating models
  util/                   the four dataset configurations, and small helpers
```

Twenty-two files, listed one by one in `sim`'s `UPSTREAM_FILES` rather than
discovered by walking the repository — a walk would also bring its agent
runners, its analysis scripts and its paper.

## What was left out

Two files upstream ships are deliberately not here. Nothing the environment
imports reaches either, and the first would put `openai` in this module's
requirements for code that never runs.

* `module/stream_chat.py` — an OpenAI client, for upstream's own LLM runners
* `module/strategy_manager.py` — the strategy document the `plan_and_act` runner
  passes between its phases

## What was added

`module/__init__.py` and `model/__init__.py`, both empty. Upstream runs from its
repository root; these make the directories importable from elsewhere. They are
listed apart from the manifest (`sim`'s `OURS`) so that a checksum mismatch
always means a vendored file was edited, never that we added one.

The modules under `module/` import their siblings bare (`from sku import SKU`),
which is why [`env.py`](env.py) puts that directory on `sys.path` as well as
`upstream/` itself.

## Two behaviours worth knowing about

Neither is changed here — a fix belongs upstream, and a workaround belongs in
`env.py` — but both shape what a run means.

* **`exec_tools` never raises.** It catches every exception and answers with a
  result carrying an `error` key, so a bad supplier, a malformed argument and
  insufficient funds all arrive the same way. `Session.act` turns them back into
  refusals.
* **Expired stock is worth nothing.** `inventory.step` sells it at
  `buy_price * 0`, though the comment beside it says 0.6×. Missing a shelf life
  costs the whole cost of the goods.

## The data is not here

It is fetched by `sim fetch`, from the same commit, into `.cache/data`. About
700 MB, most of it per-SKU supplier quote histories at six megabytes each — not
something to put in a git history, and upstream's to distribute.
