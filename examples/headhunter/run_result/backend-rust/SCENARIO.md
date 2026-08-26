# Scenario

Give the agent one job posting. It searches a pool of 600 people, picks three, and
drafts a cold mail for each one.

```
eval/jd/backend-rust.md ──▶ [ headhunter agent ] ──▶ 00-shortlist.md
                                   │                 01-<name>.md
                                   │                 02-<name>.md
                                   ▼                 03-<name>.md
                        data/headhunter.db (600 people, FTS5)
```

The agent gets one command: `headhunting`, which the app builds and attaches. It
searches, reads profiles in full, and decides what to do with what it finds. Nothing
else reaches the pool — the tree it can see holds the posting, the schema, and the
files it writes.

`cortex` mounts that tree and registers `headhunting` as a delegated executable, so
ailoy's existing `shell` tool can call it. The adapter code for this is zero lines.

## Why the pool is hard to query

600 people is not 600 random people. 65 of them are built by hand, and 17 of those are
traps: profiles that pass a reasonable query and fail a careful read.

| Trap | What it looks like | Why a query misses it |
| --- | --- | --- |
| `headline-bait` | headline says `learning Rust` | `--mentions rust` finds it and `--skill rust` does not. The headline is all there is. |
| `overlapping-tenure` | two jobs overlap by 24 months | Nothing says they overlap. Summing job lengths gives 6.0 years where the real figure is 4.0, and the posting's floor is 4. |
| `same-name` | two people, one name | Merging them invents a career. Splitting the wrong pair loses one. |
| `inflated-title` | CXO at a 10-person company | The title alone reads as the most senior person in the pool. |

Another 11 people are controls. A control differs from its trap on exactly one axis,
and on that axis it passes. Without controls, an agent could score well by discarding
anything that looks odd. The controls make that strategy visible, because it throws
them away too.

The schema stores no derived values. Total months of experience, current employer and
current title are not columns; you compute them from `positions`. Storing them would
kill two traps at once.

## What happened in one run

Run of 2026-08-21 against `eval/jd/backend-rust.md`, a Rust backend role on a payments
settlement platform. Full capture in [`run-2.console`](run-2.console).

The agent read `skill_distribution` before searching anything. Rust is spelled seven
ways in this pool, and the view says so:

```
Rust 52 · rust-lang 14 · Rust (Programming Language) 7 · Rust Lang 7
러스트 5 · Async Rust 2 · Tokio 2
```

Knowing that, it searched the full-text index instead of a single skill string, then
narrowed. Every number below is reproducible against the committed database.

| Step | Predicate | Left |
| --- | --- | --- |
| Search | `candidate_fts MATCH 'rust'` | 83 |
| Location | `city IN ('Seoul','Seongnam')` | 68 |
| Reachable and experienced | `real_years >= 4` and a row in `contacts` | 31 |
| Domain | second pass on `payment OR 정산 OR idempoten OR kafka` | read in full |
| Written | three picks, one mail each | 3 |

It ran the payments query a second time without the Rust filter, to catch anyone whose
Rust is spelled in a way the first pass missed. Then it read the full position text for
everyone with seven or more years and a settlement signal, and pulled
`candidate_tenure.real_months` rather than summing job lengths, because concurrent
roles inflate a naive sum.

### The three it picked

| | Who | Why |
| --- | --- | --- |
| 1 | 하은 성 `urn:li:person:b3tq9wmk` | Rewrote a settlement batch in Rust, cutting close time from a day to an hour, and did the PostgreSQL partitioning and read replicas. The posting asks for that exact work. |
| 2 | Reese Whitlock `urn:li:person:e8mk2wpb` | Owns a Rust card-authorization service at about 3,000 transactions per second, rewrote the settlement reconciler, built the idempotency layer for safe acquirer retries. |
| 3 | 채원 노 `urn:li:person:k6qb3nwd` | Built a Rust event-ingestion gateway at about 80,000 events per second and designed the offset-management and dedup layer so reprocessing does not duplicate. |

Two of the three are labelled `clear-fit` in the answer key. The third is a control for
the `location-mismatch` trap, which means it belongs in a shortlist and is there to
catch agents that reject people for living in the wrong place without checking where
they work.

### What it caught

The agent found three different people named 서연 강 and two named Reese Whitlock. It
listed all of them with their ids and said which one it meant. One of those pairs is
the `same-name` fixture. The other two are not registered as traps at all. The name pool
is finite, so duplicates happen by accident, and the agent treated the accidental ones
exactly as it treated the designed one.

It rejected Morgan Thorne `urn:li:person:t4qm9wbd` for having no current position and no
contact row. That profile is half of the `rank-inversion-pair` fixture, where naive
summing puts the weaker candidate ahead.

It rejected Alex Merrick `urn:li:person:r7nk4qwj`, whose headline reads "fintech
payments, web3 curious" and who surfaced on the settlement query, on the grounds that
no position description mentions Rust anywhere. The answer key marks that profile
`clear-miss`.

On Devon Grantham `urn:li:person:g9wr5tvb` it wrote:

> genuine Rust/Python backend engineer, contactable, but shorter tenure and no
> settlement/payment/Kafka signal in his one position; kept in reserve behind the picks
> above, **not a trap but not top-3**.

That person is the control for `headline-bait`. The agent ranked them below the picks
and said explicitly that they are not a trap. An agent working by elimination would
have discarded them along with the trap they control for.

### What it did not reach

`overlapping-tenure` sits on Riley Calloway `urn:li:person:v2ncq8jf`, whose skills are
Rust and Distributed Systems and whose profile contains no payments vocabulary at all.
Any agent that narrows a settlement posting by domain drops that person before reading
anyone. Two separate runs did exactly that.

The trap works. This posting simply cannot reach it, which says something about the
fixture and nothing about the agent. The scorer flags it for a human to check rather
than passing it silently.

## Output

```
run-2/backend-rust/
  00-shortlist.md      picks, method, and rejections with reasons
  01-하은-성.md          one cold mail per pick, in rank order
  02-reese-whitlock.md
  03-채원-노.md
```

Each mail is written in the candidate's `profile_language`. Two came out Korean and one
English, which the data decided, not a setting.

The shortlist has to name rejections. A list of three names cannot be checked later:
you cannot tell whether the agent filtered out the trap or never saw it. The
instruction requires a `<!-- rejected -->` marker, and the scorer reads what follows it.

## Scoring

```console
$ python3 eval/run_eval.py --score run-2/backend-rust
```

This run had no automatic failures and twelve items flagged for human review. The
scorer does not try to settle those twelve. Whether a missing control is a ranking
decision or an elimination depends on the reason the agent gave, and only reading the
rejections answers it.

## Running it

```console
$ export AILOY_CORTEX_CONSOLE=../../../cortex/target/debug/cortex-local-console
$ cd examples/headhunter
$ python3 sql/load.py
$ cargo run -p headhunter -- --jd eval/jd/backend-rust.md --out run-2 --k 3
```

The default model is `bedrock/global.anthropic.claude-sonnet-5`. Pass
`--model anthropic/claude-sonnet-5` to use the Anthropic API instead. Credentials come
from `.env` at the repository root.

Three other postings live in `eval/jd/`. One of them, `blockchain-solidity.md`, has no
qualified candidate anywhere in the pool. Saying so is the correct answer.
