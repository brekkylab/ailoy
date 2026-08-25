# Company Analysis Example

Ask a question about a company — whatever the [GLEIF][gleif] and [SEC EDGAR][edgar] APIs
can reach — and get an answer with its sources attached.

The agent researches the question against those two registries and writes back a report
in which every claim names the path it came from. It is the reading half of due
diligence: who an entity is, whether two records are the same company, who owns whom,
and — as often as not — which of those the registries decline to say.

```
   gleif/ ─┐                            artifacts/<run>/
           ├──▶ [ agent ] ──────────▶   ├─ report.md      the answer, for a person
   edgar/ ─┘         ▲                  ├─ evidence.md    every claim ↔ its path
                     │                  ├─ findings.json  the same, for a machine
                  question              └─ queries/*.py   what was actually run
```

The agent gets the built-in file tools and nothing else — no bespoke search tool, no
client library — and still every answer is current, because the tree it opens is served
by the registries' own APIs rather than by a copy of them.

[gleif]: https://www.gleif.org/en/lei-data/gleif-api
[edgar]: https://www.sec.gov/edgar/sec-api-documentation
[fs]: https://github.com/brekkylab/cortex/blob/main/cortex/src/fs/filesystem/filesystem.rs
[lei]: https://www.gleif.org/en/organizational-identity/lei-vlei/the-legal-entity-identifier-lei
[cik]: https://www.sec.gov/search-filings/cik-lookup
[xbrl]: https://www.sec.gov/data-research/standard-taxonomies

---

## 1. What is mounted

**Nothing here is stored.** Each registry is a cortex [`FileSystem`][fs]: every
filesystem operation — listing a directory, checking whether a path exists, reading a
file's bytes — is answered by the GLEIF or SEC EDGAR API. A path that looks like a file
is a call going out, and a listing is what the registry says at that moment. Two runs a
day apart can differ, and that is the registry disagreeing with itself rather than the
tool.

| | GLEIF | SEC EDGAR |
| --- | --- | --- |
| Covers | legal entities worldwide, and who owns whom | anyone required to file with the SEC |
| How many | 3.4M LEI records | 982k filers — funds, foreign issuers, subsidiaries, insiders; about 8k of them trade |
| Identifier | LEI | CIK |
| A search returns | the whole record | a stub of three fields |
| Auth | none | a `User-Agent` naming a contact |

The two are shaped oppositely, which is most of what makes having both worthwhile. A
GLEIF search page ends the trip; an EDGAR search only tells you where to look next.

---

## 2. Mounted directory structure

```
<mountpoint>/
├── CATALOG.md                    written here — the one path no store answers for
├── gleif/
│   ├── CATALOG.md
│   ├── by-lei/
│   │   ├── _README.md
│   │   └── <LEI>/                ls → the resources this record links to
│   │       ├── record/
│   │       │   └── record.json
│   │       └── direct-parent/    or direct-parent-reporting-exception/
│   └── search/<field>/<value>/…/
│       └── pages/
│           ├── _README.md       how many pages this query has
│           └── page-001.json
└── edgar/
    ├── CATALOG.md
    ├── by-cik/
    │   ├── _README.md
    │   └── <CIK>/
    │       ├── submissions/
    │       │   └── submissions.json
    │       ├── facts/
    │       │   └── facts.json
    │       └── concept/<taxonomy>/<tag>.json
    └── search/<parameter>/<value>/…/pages/
```

None of it exists until it is asked for. The shape is fixed and known here; the contents
arrive from the API on the read that names them.

Everything in angle brackets is a name you supply, and they come in three kinds.

### 2.1 Identifiers

Each names one company, and which one you hold decides which door you come in by. The
definitions belong to the registries and are linked rather than restated.

| | | Where one comes from |
| --- | --- | --- |
| `<LEI>` | [Legal Entity Identifier][lei] | a `gleif/search` page, or `owns` / `ownedBy` on another record |
| `<CIK>` | [Central Index Key][cik] | an `edgar/search` hit, or `submissions.json` naming its own |

The LEI and the CIK are checked here rather than by the API: an LEI carries ISO 17442
check digits and a CIK is at most ten digits, so a mistyped one is a directory that does
not exist — a `No such file` before anything goes out. The CIK has three spellings,
`1045810`, `0001045810` and `CIK0001045810`, all naming the same registrant.

What that check cannot catch is an identifier that is well formed and wrong: that one
answers with somebody else's company rather than with an error. Identifiers are worth
confirming rather than recalling.

### 2.2 Query segments

A filter is a pair — a `<field>` and its `<value>` on GLEIF, a `<parameter>` and its
`<value>` on EDGAR — and pairs stack by descending:

```
gleif/search/entity.legalAddress.country/KR/entity.status/ACTIVE/pages/
edgar/search/forms/10-K/entityName/Samsung/pages/
```

Upstream those are `filter[entity.legalAddress.country]=KR&filter[entity.status]=ACTIVE`
and `forms=10-K&entityName=Samsung`. Order does not matter: either way round is one
query and one cached result.

`ls search` names the fields on offer — twelve on GLEIF, six on EDGAR — so **a wrong
field is a missing directory rather than a failed request.** GLEIF accepts more than
twelve; the rest are vendor cross-references and registry bookkeeping, left out because
no question about a company needs them. Values are not listable and could not be; you
write those.

None of this has asked the API anything. The first request happens when `pages` is
opened, because only the API knows how many results there are. That listing then names
one page and a note: a directory read costs a `getattr` per entry and a page's size is
unknowable without fetching it, so naming every page would spend a request each before
anything was read. `page-002` onwards still open, unlisted — and the note says how many
there are, taken from the page the listing has already fetched, so learning the count
costs neither a request nor a screenful of records.

### 2.3 Concept coordinates

A `<taxonomy>` and a `<tag>` address one XBRL number over the whole filing history:

```
edgar/by-cik/0001045810/concept/us-gaap/Revenues.json
edgar/by-cik/0001045810/concept/dei/EntityCommonStockSharesOutstanding.json
```

The five taxonomies are [the ones the SEC publishes][xbrl] — `us-gaap`, `dei`,
`ifrs-full`, `srt`, `invest` — and `ls concept` names them. The tags are that company's
own: whichever it reports, which is the keys of its `facts.json`.

### 2.4 `_README.md`

Some directories do not list what they hold. `by-lei` cannot: the API has no index to
build one from. `by-cik` could — the SEC publishes all 982,172 of them — but a million
zero-padded numbers is a listing nobody can read a company out of, so it does not.
Either way, answering empty would read as "nothing here", so each holds a `_README.md`
saying how to address one and where an identifier comes from.

They are not the exception here — no directory in this tree can be walked into. Where a
listing *is* complete it is also small and fixed: the taxonomies under `concept`, the
fields under `search`, and, under a GLEIF entity, that record's own `relationships`, so
a company with no parent shows a reporting exception where one with a parent shows a
parent.

### 2.5 Why `<name>/<name>.json` instead of `name.json`

Every fetchable document sits inside a directory of its own, holding one file named
after it. The reason is that **files cost and directories do not.** A directory read is
followed by a `getattr` per entry, a document's size cannot be known without fetching
it, and so a document listed directly is fetched merely to be listed.

Wrapped, `ls by-cik/<CIK>/` names `submissions/`, `facts/` and `concept/` for nothing,
and the request waits until someone descends. Listed flat, that same `ls` costs both
documents — `facts.json` is four megabytes of it — whether or not the caller wanted
either.

Nothing is hidden by this. The flat `submissions.json` beside the wrapper opens too, in
the same way `page-002` does. Not listed is not unreachable.

---

## 3. Usage

```sh
# 1. Credentials — the repository's .env is read at startup
ANTHROPIC_API_KEY=...
SEC_USER_AGENT='company-analysis you@example.com'

# 2. A libfuse provider, and the console server
brew install --cask fuse-t
export PKG_CONFIG_PATH="$PWD/../cortex/cortex/contrib/pkgconfig:/usr/local/lib/pkgconfig"
cd ../cortex && cargo build -p cortex-local-console
export AILOY_CORTEX_CONSOLE=$PWD/target/debug/cortex-local-console

# 3. Ask
cd examples/company_analysis
cargo run -p company_analysis -- --preset entity-profile --company "Samsung Electronics"
cargo run -p company_analysis -- --task "Who owns Alphabet's subsidiaries?"
```

`SEC_USER_AGENT` is required rather than defaulted: SEC answers 403 with an HTML page to
a `User-Agent` that names no contact, and a default would only move the failure to the
first read, where it arrives as a JSON syntax error.

`PKG_CONFIG_PATH` points at a shim cortex ships, because FUSE-T installs `fuse-t.pc`
while the mount bindings probe for `fuse.pc`.

### Presets

Three questions come ready-made. Each takes `--company <name>`, and each fixes the body
of the report — the **Summary**, **Data limits** and **Next steps** are common to all
three.

| | Asks | The report body |
| --- | --- | --- |
| `entity-profile` | who is this? | the entity · identifiers and where each comes from · registration and status · what the filings say |
| `cross-registry` | is this the same company in both? | the two records · how they were matched and how sure that is · where they agree · where they disagree |
| `ownership-tree` | who owns whom? | the group as disclosed · direction of each relationship · where the tree stops · what the shape does not tell you |

The answer to any of them may be that the registries do not carry it, and saying so is
treated as an answer rather than a failure.

`--task` takes a question in your own words instead, and the body is then organised to
fit it. `--since <findings.json>` makes the report cover what changed against a previous
run.

---

## 4. Deliverables

One directory per run, and the four files answer four different readers:

| | Answers | Shape |
| --- | --- | --- |
| `report.md` | what was found | a **Summary** that stands alone, then **Data limits** and **Next steps** |
| `evidence.md` | how it is known | one bullet per claim, naming the path and the field |
| `findings.json` | the same, for a machine | fixed schema, so two runs diff |
| `queries/*.py` | what was actually run | numbered, so the route to a finding can be walked again |

`findings.json` carries `run_id`, `task`, `entities`, `data_gaps`, and a `findings` list
of `{severity, statement, evidence, confidence}`.

Because the source is live, a claim is only as good as the path behind it.
`evidence.md` cites from the mount root down — `edgar/by-cik/0001045810/…` — never the
machine-local prefix, which would leave the citation unresolvable anywhere else.
`findings.json` is what `--since` reads back in, so a later run can report what changed
rather than restate what did not.

---

## 5. The run summary

```
turns 61 / tool calls 32
tokens (29 calls): in 675,505 / out 20,540
  largest context 43,746 (the conversation at its last call)
requests  gleif 5 / edgar 2
  gleif   linked resource ×2, search page ×2, lei record ×1
          5 distinct paths; heaviest:
               1 × /by-lei/549300S4KLFTLO7GSQ80/record.json
  edgar   facts ×1, submissions ×1
finish    Stop
artifacts 5:
  ...
```

`requests` is the number that matters for the design: narrowing should not move it, and
a large number means the agent paged through a broad query instead of asking a narrow
one. The breakdown beneath it is what makes that readable — a total says a run was
expensive without saying what it bought, and the distinct-path count separates one
document fetched many times from many fetched once. Each tool call is printed with the
requests it cost, so the answer is attributed rather than inferred.

---

## 6. Limits

Three things this example does not have.

**A reliable key between the registries.** `submissions.json` has an `lei` field and it
was empty for all fifteen registrants sampled; EIN does not bridge either. Crossing
between them means matching on a name, and a name match is a candidate — a search for
NVIDIA returns a fund before it returns the manufacturer. The instruction requires such
a match to be reported as unconfirmed.

**Sanctions, litigation, news, or financial statements outside the US.** Questions
needing those have no answer here, and the instruction treats saying so as the answer.

**Enforcement of the write boundary.** The stores refuse writes, but commands run on the
host and the mountpoint's parent is an ordinary directory. `guard.rs` detects a write
that left the allowed directories; it does not prevent one.

---

## 7. Layout

```
examples/company_analysis/
  README.md
  Cargo.toml
  src/
    main.rs      CLI, the two mounts, the run summary
    prompt.rs    system instruction and presets — policy, not the map
    apifs.rs     the client, cache and request count both stores share
    gleif.rs     GLEIF as a FileSystem
    edgar.rs     EDGAR as a FileSystem
    guard.rs     which directories a run may write into
  artifacts/     one directory per run
  workspace/     scratch
```

Each store's tests cover its path grammar offline; the ones that mount and talk to a
registry are `#[ignore]`d.

```sh
cargo test -p company_analysis
SEC_USER_AGENT='...' cargo test -p company_analysis -- --ignored --test-threads=1
```

`--test-threads=1` is not optional for the mounting ones. Each holds a mount for the
length of the test and unmounts by dropping it, and two of those overlapping deadlock —
the tests report their results and then hang on the way out.
