# Headhunter Example

A headhunting agent whose environment is a talent-pool database.

Given a job description, the agent searches the pool, picks the top-k candidates, and drafts a personalized cold mail for each one under `artifacts/`.

```
jd.md ──▶ [ headhunter agent ] ──▶ artifacts/<slug>/00-shortlist.md
              │                    artifacts/<slug>/01-<candidate>.md
              │                    artifacts/<slug>/02-<candidate>.md
              ▼                    ...
        candidate DB (mock)
```

---

## 1. Scenario

1. **Talent pool** — a local database of profiles, assumed to have been collected from LinkedIn. The schema mirrors the LinkedIn Profile API (Profile / Positions / Educations / Skills).
2. **Input** — a single job posting, given as `jd.md`.
3. **Agent** — holds the DB as its environment, reads the JD, searches and evaluates candidates, and writes a cold-mail draft for each of the top-k into `artifacts/`.

The agent has **read-only** access to the DB. The only place it writes is `artifacts/`.

---

## 2. Scope

**In**

- A mock candidate DB with a LinkedIn-like schema (JSON seed + query tools)
- A single agent loop: parse JD → search → rank → draft cold mails
- A reproducible artifact directory per run

**Out (later)**

- Real LinkedIn API integration or scraping
- Actually sending mail, handling replies, managing sequences
- Embedding-based vector search (the `ailoy` crate has no embedding API today — see §5)
- ATS/CRM integration, batch processing of multiple JDs

---

## 3. Data: the candidate DB

### 3.1 Storage

- Seed data lives in `data/candidates.json` — an array of `Candidate`.
- Size: 30–50 profiles to start. The set should mix clear fits, clear misses, and
  borderline cases, so ranking quality can be judged by eye.
- Access goes behind a `CandidateDb` trait. The mock implementation loads the JSON
  into memory; a real database implementation can be swapped in behind the same trait.

### 3.2 Schema (LinkedIn-style)

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

## 4. Input: `jd.md`

An ordinary markdown job posting. No front matter is required — structuring it is the
agent's job. One sample `jd.md` ships with the example.

It should carry:

- Company / team / role title
- Core responsibilities
- Must-have and nice-to-have requirements
- Location and employment type
- (optional) Salary range, recruiter name and contact — used in the mail signature

The path is taken as a CLI argument, defaulting to `./jd.md`.

---

## 5. The agent

### 5.1 Environment = DB query tools

Read-only tools exposed to the agent (registered via `ToolProvider::insert_func`):

| Tool | Input | Output |
| --- | --- | --- |
| `search_candidates` | `keywords[]`, `titles[]`, `skills[]`, `location`, `min_years_experience`, `open_to_work`, `limit` | Matching candidate summaries (id, name, headline, current title/company, years of experience, top skills) |
| `get_candidate` | `id` | The full profile JSON for that candidate |
| `db_stats` | — | Total headcount, distribution of top skills / titles / locations |

- `search_candidates` only narrows the pool, via filters plus keyword scoring.
  The final judgment is the agent's, made after reading full profiles with `get_candidate`.
- Every response includes `total_matched`, so the agent can tell when results were truncated.
- `db_stats` lets the agent tune its queries — e.g. if "Rust" matches nobody, widen to
  adjacent skills.

**Why not embedding search**: the `ailoy` crate currently exposes no embedding or vector
store API. So "similarity" is implemented as structured filtering to build a candidate
set, followed by LLM ranking with stated reasons. If embeddings land later, only the
internals of `search_candidates` need to change.

### 5.2 Writing artifacts

Writes under `artifacts/` use the built-in `write` tool. The agent instruction constrains
paths to `artifacts/<run-slug>/`.

### 5.3 Loop

1. Read `jd.md` and structure the requirements (must-have / nice-to-have, titles,
   skills, location, years).
2. Call `db_stats` to understand the shape of the pool.
3. Issue several `search_candidates` queries to assemble a candidate set (roughly 3k–5k
   people). Not one query — split by title, by skill, and by adjacent skill.
4. Read each candidate's full profile with `get_candidate` and evaluate them.
5. Select the top **k** (default `k = 3`, overridable by CLI argument).
6. Write one shortlist document plus k cold-mail drafts into `artifacts/`.

### 5.4 Ranking requirements

- Every candidate gets a 0–100 score and a **rationale**. The rationale cites only facts
  present in the profile (company, title, tenure, skills).
- Risks and mismatches are recorded too (location mismatch, insufficient tenure,
  different domain).
- A candidate who clearly fails a must-have requirement does not make the top k.
- If fewer than k candidates qualify, **emit fewer and say why in the shortlist.**

---

## 6. Output: `artifacts/`

```
artifacts/
  2026-08-16-acme-senior-backend/
    00-shortlist.md
    01-jane-doe.md
    02-minsu-park.md
    03-alex-kim.md
```

Directory name: `<date>-<company-slug>-<role-slug>`.

### 6.1 `00-shortlist.md`

- Summary of the JD (the structured requirements)
- The search queries used and how many each matched
- A table of the selected k: name, current title/company, score, one-line rationale,
  profile URL
- Two or three near-misses whose rejection reason is informative
- Items a human must check (e.g. no contact info, uncertain tenure estimate)

### 6.2 Per-candidate draft `NN-<slug>.md`

```markdown
---
candidate_id: urn:li:person:aBcD1234
candidate_name: Jane Doe
profile_url: https://www.linkedin.com/in/jane-doe-1234
score: 87
subject: "Acme payments platform — Rust backend role"
---

## Why this candidate

- (up to three reasons, each grounded in profile facts)

## Mail body

Hi Jane,
...

## Needs review

- (what a human should confirm before sending)
```

Requirements for the mail body:

- Written in the candidate's likely working language — the agent decides from the
  profile (English by default, Korean when the profile is Korean-facing)
- 200–250 words: greeting, one personalized paragraph, role summary, call to action
- **The personalized sentence must be grounded in a fact from the profile.** Guesses
  about salary, willingness to move, or job satisfaction are never stated as fact.
- Unconfirmed figures (compensation) are cited only from the range given in the JD
- It must read as a draft, and presumes human review before anything is sent

---

## 7. Running it

```sh
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, etc.

cargo run -p headhunter -- \
  --jd ./jd.md \
  --db ./data/candidates.json \
  --top-k 3 \
  --out ./artifacts
```

- The model is set with `--model`, defaulting to `anthropic/claude-sonnet-4-6`.
- Progress streams through the `cortex` console.
- On exit, the paths of the generated files are printed.

---

## 8. Layout

```
examples/headhunter/
  README.md
  Cargo.toml
  src/
    main.rs        # CLI, agent assembly, run
    db.rs          # CandidateDb trait + JSON mock implementation
    schema.rs      # LinkedIn-style types
    tools.rs       # search_candidates / get_candidate / db_stats
    prompt.rs      # system instruction
  data/
    candidates.json
  jd.md
  artifacts/       # gitignored (one sample run committed)
```

`examples/headhunter` is added to `[workspace] members` in the root `Cargo.toml`.

---

## 9. Done when

- [ ] `cargo run -p headhunter` works with nothing but an API key set
- [ ] The bundled `jd.md` produces one shortlist and k mail drafts
- [ ] Personalized sentences in the drafts match the actual profile (verified by hand)
- [ ] No invented candidates, companies, or career history
- [ ] `CandidateDb` sits behind a trait, replaceable by a real database

---

## 10. Open questions

1. Keyword scoring in `search_candidates` — start with plain substring matching, or go as far as BM25?
2. Evaluate candidates in sub-agents (one per candidate) or in the main loop? Sub-agents save context but make the example harder to follow.
3. Should the schema carry a contact email? The LinkedIn API does not expose one, so framing the output as an "InMail draft" and omitting the field is more realistic.
4. Do generated artifacts get committed to the repository?
