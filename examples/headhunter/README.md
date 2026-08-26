# Headhunter Example

A headhunting agent whose environment is a talent-pool database, reached through a command
this app attaches to the console.

Given a job posting, the agent searches a pool of 600 candidates, picks the top k, and
drafts a cold mail for each one.

```
jd/<posting>.md ──▶ [ headhunter agent ] ──▶ <out>/<posting>/00-shortlist.md
                           │                 <out>/<posting>/01-<candidate>.md
                           │                 <out>/<posting>/02-<candidate>.md
                           ▼                 ...
                headhunting search / read
                  distribution / query
                           │
                           ▼
                data/headhunter.db  (read-only)
                           ▲
                     sql/load.py
                           │
                data/candidates.json  (committed)
```

**The database is not in the repository — `python3 sql/load.py` builds it** from the
committed JSON, and that is the first thing to run. A binary that large does not diff, and
it is a second's work to regenerate, so what is kept is what it is built from.

The pool is read-only, and the only place the agent writes is the run's own directory.
A real run of all four postings is committed under `run_result/`, so the example can be
read without being run.

---

## The pool

### Where it lives

- Seed data lives in `data/candidates.json` — an array of `Candidate`. `sql/load.py`
  loads it into `data/headhunter.db`, which is what the agent actually queries.
- Size: 600 profiles, mixing clear fits, clear misses, and borderline cases, so ranking
  quality can be judged by eye.
- Access goes through the `headhunting` executable (below), which opens the database
  read-only. Swapping in another store means reimplementing that command, not the app.

### Schema (LinkedIn-style)

Field names follow LinkedIn Profile API conventions (camelCase).

```jsonc
{
  "id": "urn:li:person:aBcD1234",
  "publicProfileUrl": "https://www.linkedin.com/in/jane-doe-1234",
  "firstName": "Jane",
  "lastName": "Doe",
  "headline": "Senior Backend Engineer @ Acme · Rust, Distributed Systems",
  "summary": "Backend engineer with 8 years of experience in high-traffic systems …",
  "location": { "country": "KR", "city": "Seoul" },
  "industry": "Computer Software",
  "positions": [
    {
      "title": "Senior Backend Engineer",
      "companyName": "Acme",
      "companyUrn": "urn:li:organization:9876",
      "employmentType": "FULL_TIME",
      "location": "Seoul, KR",
      "startDate": { "year": 2022, "month": 3 },
      "endDate": null,               // null means currently employed
      "description": "Payment settlement pipeline …"
    }
  ],
  "educations": [
    {
      "schoolName": "Seoul National University",
      "degreeName": "B.S.",
      "fieldOfStudy": "Computer Science",
      "startDate": { "year": 2013 },
      "endDate": { "year": 2017 }
    }
  ],
  "skills": [{ "name": "Rust", "endorsementCount": 24 }],
  "certifications": [{ "name": "AWS SA Associate", "authority": "AWS" }],
  "languages": [{ "name": "Korean", "proficiency": "NATIVE_OR_BILINGUAL" }],
  "openToWork": true,
  "connectionsCount": 780,
  "lastUpdatedAt": "2026-05-02"
}
```

Derived values (total months of experience, current employer/title) are not stored —
they are computed from `positions`.

> Privacy: all seed data must be synthetic and unrelated to real people.
> Any email address uses the `example.com` domain.

---

## The posting

An ordinary markdown job posting. No front matter, no fixed headings — structuring it is
the agent's job. Four ship with the example, under `jd/`, and each one leans on a
different part of the pool:

| posting | k | what it asks for |
| --- | --- | --- |
| `backend-rust.md` | 3 | Rust and Seoul, written in Korean |
| `backend-seoul-ko.md` | 3 | the same market, where the mail's language is the test |
| `ml-platform-tokyo.md` | 12 | more people than the pool can supply |
| `blockchain-solidity.md` | 5 | a skill nobody in the pool has |

The path is a CLI argument. What it points at is copied into the run's tree as `in/jd.md`,
so the posting the agent read is kept beside what it produced.

---

## The agent

### The environment is an executable the app attaches

**This is what the example is for.** The app builds a command in its own domain, registers
it with the cortex console, and that command is the agent's only way to reach the pool.
Instead of writing SQL, the agent asks in the vocabulary of recruiting.

| Command | Input | Answer |
| --- | --- | --- |
| `headhunting search` | `--skill` `--mentions` `--city` `--min-years` `--name` `--id` | A table, one line per person |
| `headhunting read` | several ids | A block: profile, tenure, position descriptions in full, skills, contacts |
| `headhunting distribution` | an axis (`skill`, `city`) and a term | what values that axis is made of |
| `headhunting query` | read-only SQL | A table |

`search` builds the candidate set; `read` is how the agent judges it. `query` is the way
out when neither of the other two can express the question — sorting the 29 queries of a
real run by kind, not one of them needed free-form SQL.

What it actually looks like. Every call a run made is in that run's `queries.log`, with
the command text untruncated.

```
$ headhunting search --skill rust --city Seoul,Seongnam --min-years 4 --limit 3
skill "rust" -> Rust(52) rust-lang(14) Rust (Programming Language)(7) Rust Lang(7) Async Rust(2)  ·  the pool holds 49 skill spellings (--spellings for all of them)

id                      name            years  open  contact  headline
urn:li:person:b2pq7x72  예은 연         15.0   n     0        Sr. Backend Engineer · Large-scale Distributed Systems, PostgreSQL, Rust
urn:li:person:bqsuo43r  Casey Dunmore   14.0   n     0        Sr. Backend Engineer · rust-lang and Kafka
urn:li:person:0s3tprad  Devon Grantham  12.0   n     0        Senior Backend Engineer · Python, Distributed Systems, Java
-- 3 of 52 rows
```

`years` has concurrent employment merged; `contact 0` means there is no way to reach
this person. Both are material for the judgment, not conditions that remove anyone.

The command answers its own usage (`headhunting <command> --help`), so the instruction
does not repeat it.

**`--skill` and `--mentions` look at different things.** The first reads only the skill
list; the second also reads the headline, summary, titles, and position descriptions.
Someone who put "learning Rust" in their headline and nowhere else is found only by the
second.

**What decides is shown, not filtered.** Open-to-work, whether the person can be
contacted, and real years of experience are columns in the table, but no condition
removes anyone by them. Filter them away in the tool and the agent never learns that a
strong candidate who isn't looking existed — and cannot say in the shortlist why they
were passed over.

**Spelling drift is surfaced, not absorbed.** The first line of the answer says which
spellings this search caught and how many spellings the pool holds. The same skill is
written as `Rust` and as `러스트`, and the second shares no characters with the first, so
it is not found. Whether to widen is the agent's call, made after `headhunting distribution skill`.

**Why not embedding search**: the `ailoy` crate currently exposes no embedding or vector
store API, and measured coverage and ranking matched FTS5.

### Writing artifacts

Writes use the built-in `write` tool. **The tree the agent sees is this run's artifact
directory** — the instruction does not constrain paths; there is nothing outside it.

```
<out>/<jd>/          the root of the tree and where artifacts are written
  in/jd.md           the posting
  in/schema.sql      table definitions
```

**The view definitions are not handed over.** Every view is behind a command:
`candidate_tenure` and `current_position` are folded into `search` and `read`,
`candidate_brief` is what `search` stands on, and the distributions are `distribution`'s
axes. Across three runs, 87 free-form SQL statements touched a view five times, and all
five asked for figures `read` already carries.

`data/` is outside the tree. It holds the answer key (`ground_truth.json`, saying which
of the 600 are planted and what each one tests) and the full source pool
(`candidates.json`); mount the whole
directory and one `read` call reaches all of it. The pool database is outside too, opened
by the host path `headhunting` was given at registration. That is why the command line
carries no db argument.

**This is not a wall.** Only the file tools (`read`, `write`, `glob`, `grep`) are confined
to the tree. `shell` is not: `cortex-local-console` spawns `sh -c` on the host and only
sets `current_dir` to match the session, so an absolute path goes anywhere — in a real run
the agent wrote intermediate results to `/tmp` and those files stayed on the host.

Narrowing the tree therefore narrows the *default* path; building a wall would require the
console server to spawn the shell inside an isolated filesystem, which is not this
example's call. So instead we **check whether it happened**: at the end of a run every
shell command is scanned, and if any reached the answer key or the full pool the run
fails. Not knowing whether the shortlist was reasoned to or looked up is worse than a
quiet pass.

### The loop

1. Read the posting and **separate must-haves from nice-to-haves.**
2. Read `in/schema.sql` for the table definitions. The view definitions are not in the
   tree — the arithmetic they hold is already in what `search` and `read` answer.
3. Gate with `headhunting search`, **on the must-haves only** — put the domain vocabulary
   in the conditions and a qualified person who never uses those words disappears before
   anyone reads them.
4. Read the first line of the answer and decide whether to widen.
5. Check nice-to-haves **within** what was found (`search --id … --mentions …`). This
   ranks; it does not narrow.
6. `headhunting read` several people **at once**. Some things are only visible side by side.
7. Select the top **k** (default `k = 3`).
8. Write one shortlist plus k cold-mail drafts.

### Ranking rules

- Every candidate gets a **rationale**, citing only facts present in the profile
  (company, title, tenure, skills). No score — compressing the judgment into one number
  hides what it was made of.
- **Name the people who were rejected, and why**, especially anyone a naive query would
  have ranked highly. A shortlist without its rejections cannot be checked later.
- Risks and mismatches are recorded too (location mismatch, insufficient tenure,
  different domain).
- A candidate who clearly fails a must-have requirement does not make the top k.
- If fewer than k candidates qualify, **emit fewer and say why in the shortlist.**

---

## What comes out

One shortlist, and one cold mail per pick, in a directory named after the posting.

```
run_result/backend-rust/
  00-shortlist.md       who was picked, how, and who was rejected
  01-하은-성.md          one mail per pick, in rank order
  02-채원-노.md
  03-도윤-반.md
  queries.log           every pool call, untruncated
  console.txt           what went by on screen
  SCENARIO.md           written afterwards: which axis of the pool this posting tests
```

**`run_result/` holds a real run of all four postings**, kept so the example can be read
without being run. It is what `cargo run` wrote, less the `in/` directory the run is
given: that is a copy of `jd/<posting>.md` and `sql/schema.sql`, made fresh on every run,
so keeping it here would be the same bytes a second time. The paths are rewritten where
they named the machine it ran on — the two host paths in the header, and the posting and
output directories, which were `eval/jd/` and `--out run-8` at the time.

| posting | k | picked | turns | pool calls | output |
| --- | --- | --- | --- | --- | --- |
| `backend-rust` | 3 | 3 | 42 | 11 | 31.7K |
| `backend-seoul-ko` | 3 | 3 | 31 | 9 | 30.5K |
| `ml-platform-tokyo` | 12 | **6** | 37 | 10 | 40.5K |
| `blockchain-solidity` | 5 | **0** | 30 | 10 | 8.4K |

The two bold figures are the point of those two postings. `ml-platform-tokyo` asks for 12
where fewer than 12 qualify — and among the near-misses is one person holding two
accounts, under two spellings of one company name. The run emitted 6 and wrote down why
it could not reach 12. `blockchain-solidity` asks for a skill nobody has, and its
shortlist opens with `## Picks` / `(none — see below)` followed by what was searched for.
Filling either number would be the failure.

### `00-shortlist.md`

Opens with a `## Picks` list, one line per selection, each carrying the full
`urn:li:person:…` — 283 of the 600 share a name with someone else, so a name does not
identify anyone. Below it: how the search was gated and widened, what was compared, and
after a `<!-- rejected -->` line, the people who were passed over and why. That marker is
what lets a reader tell a pick from a rejection.

### `NN-<slug>.md`

The candidate's id alone on the first line, then the mail. It is written in that
candidate's `profile_language`, whatever else the record says about the languages they
speak: `backend-seoul-ko` picked two `ko` profiles and one `en`, and wrote two Korean
mails and one English one. No front matter, and no score — a number hides what the
judgment was made of, and nothing reads it.

---

## Running it

The pool is loaded into SQLite once, from the committed JSON:

```sh
python3 sql/load.py            # writes data/headhunter.db
```

Then, from `examples/headhunter`:

```sh
cargo run -p headhunter -- --jd jd/backend-rust.md --k 3 --out run-1
```

- Credentials come from the repository's `.env`. The default model is
  `bedrock/global.anthropic.claude-sonnet-5`, which reads `AWS_BEARER_TOKEN_BEDROCK` and
  `AWS_REGION`; `--model anthropic/claude-sonnet-5` reads `ANTHROPIC_API_KEY` instead.
  The run in `run_result/` was made with the latter.
- `--db` points at the SQLite file and defaults to `data/headhunter.db`. It is a host
  path, not a name inside the tree the agent sees.
- `--console` is the cortex console server, built from the sibling checkout.
- Progress streams through that console; the artifact paths are printed on exit.

**Reading the result is a person's job.** Whether a personalized sentence is grounded in
the profile or reads as filler is not something to automate. What the run leaves behind to
judge it by is the shortlist's rejections, `queries.log`, and `data/ground_truth.json`,
which says which of the 600 are planted and what each one tests. `run_result/` holds one
such set to compare against.

---

## Layout

```
examples/headhunter/
  README.md
  Cargo.toml
  src/
    main.rs            # CLI, agent assembly, the run
    prompt.rs          # the system instruction
    trace.rs           # the screen, the query log, bypass detection
    executable/        # `headhunting` — the commands this app attaches
  sql/                 # schema, views, loader
  jd/                  # the four postings
  data/                # candidates.json, ground_truth.json, narration.json
                       #   headhunter.db is built from these, not committed
  run_result/          # a committed run of all four postings
    backend-rust/
      SCENARIO.md      # which axis of the pool this posting tests
```

`examples/headhunter` is added to `[workspace] members` in the root `Cargo.toml`.
