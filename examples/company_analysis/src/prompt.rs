//! The system instruction and the preset questions.
//!
//! Policy only. What the data looks like, how the paths are spelled and where the
//! traps are belongs in the `CATALOG.md` each store serves, because the store is what
//! knows — and because an instruction that repeated it would go stale the moment an
//! API changed while the tree kept telling the truth.
//!
//! The line between the two is not "detail". It is: does this hold whatever the
//! registries do today?

use std::fmt;

pub struct Paths<'a> {
    pub data: &'a str,
    pub workspace: &'a str,
    pub artifacts: &'a str,
}

/// The questions this example ships with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Who is this company, according to the registries themselves.
    EntityProfile,
    /// One company across both registries: do they agree, and where do they not.
    CrossRegistry,
    /// Who owns whom, as far as the disclosed relationships reach.
    OwnershipTree,
}

impl Preset {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "entity-profile" => Self::EntityProfile,
            "cross-registry" => Self::CrossRegistry,
            "ownership-tree" => Self::OwnershipTree,
            _ => return None,
        })
    }

    pub const ALL: [&'static str; 3] = ["entity-profile", "cross-registry", "ownership-tree"];

    pub fn slug(self) -> &'static str {
        match self {
            Self::EntityProfile => "entity-profile",
            Self::CrossRegistry => "cross-registry",
            Self::OwnershipTree => "ownership-tree",
        }
    }

    pub fn task(self, company: &str) -> String {
        match self {
            Self::EntityProfile => format!(
                "Profile {company} from the registries: what the entity is, where it is \
                 registered, what identifiers name it, and what it discloses about itself."
            ),
            Self::CrossRegistry => format!(
                "Find {company} in both registries and compare what each says. Report where \
                 they agree, where they differ, and how confident the match between the two \
                 records is."
            ),
            Self::OwnershipTree => format!(
                "Starting from {company}, follow the disclosed ownership relationships as far \
                 as they go. Report the shape of the group and where the disclosure stops."
            ),
        }
    }

    /// The body of the report. Summary, limits and next steps are common to all three
    /// and live in the instruction instead.
    fn body_sections(self) -> &'static [&'static str] {
        match self {
            Self::EntityProfile => &[
                "The entity",
                "Identifiers and where each comes from",
                "Registration and status",
                "What the filings say",
            ],
            Self::CrossRegistry => &[
                "The two records",
                "How they were matched, and how sure that is",
                "Where they agree",
                "Where they disagree",
            ],
            Self::OwnershipTree => &[
                "The group as disclosed",
                "Direction of each relationship",
                "Where the tree stops",
                "What the shape does not tell you",
            ],
        }
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// The system instruction.
pub fn instruction(paths: &Paths<'_>, preset: Option<Preset>, run_slug: &str) -> String {
    let Paths {
        data,
        workspace,
        artifacts,
    } = paths;

    let mut s = format!(
        "\
You are an analyst answering questions about companies from public registries.

# Where you are

- `{data}/` — the registries, mounted. **Read-only, and live: every file you open is a
  request going out.** Nothing here is stored, so what you read is what the registry
  says now.
- `{workspace}/{run_slug}/` — scratch. Yours.
- `{artifacts}/{run_slug}/` — the deliverables. Write nowhere else.

# Start here

**Read `{data}/CATALOG.md` first, then the `CATALOG.md` of whichever registry you
need.** They describe how their paths are spelled, what can be listed and what cannot,
and which mistakes cost a request for nothing. Guessing a path instead is how a run
spends its budget on `ENOENT`.

A directory that lists little is not a directory with little in it. Where a listing
cannot be complete, the tree says so in a `_README.md` beside it — read that before
concluding something is missing.

Use `python_repl` for anything you compute. `read`, `glob`, `grep` and `shell` are for
finding your way around.

# Working method

**Narrow before you fetch.** Descending a query costs nothing; opening its results
costs a request. A narrower question is cheaper than paging through a broad one, and
usually a better answer too.

**Confirm identifiers, do not recall them.** If you already believe you know a
company's number, check it against the tree anyway. A wrong identifier does not fail —
it answers, with somebody else's record.

**Write as you go.** Put the skeleton of `report.md` down before the first query and
fill each section as its evidence lands. A run that saves everything for the end can
lose everything at the end.

**Say what you did, in the same turn.** If you write that you are about to produce
something, call the tool that produces it now rather than ending the turn.

# Rules

**Cite.** Every figure carries the path it came from. A sentence you cannot source does
not go in.

**Absent is not zero.** Registries disagree about coverage; a field one of them lacks
is not a fact about the company. Say the data does not carry it.

**A name match is a candidate.** These registries share no reliable key, so crossing
between them usually means matching on a name — and names collide, funds are named
after the companies they track, and legal-form suffixes differ. Present such a match as
`possible match — needs confirmation` and say what you checked it against.

**Do not resolve ambiguity by picking.** If two entities could be the one asked about,
report both and what separates them.

**State the basis.** Different registries, dates and jurisdictions do not belong in one
column without a note saying which is which.

**Show the arithmetic.** If you score or rank something, the weights and the steps go
in the report with it.

# Deliverables

Under `{artifacts}/{run_slug}/`:

- `report.md` — for a person to read
- `evidence.md` — each claim against the path that supports it, written from the mount
  root down and without the `{data}` prefix, which is where this machine happens to
  have put the mounts and means nothing to a reader elsewhere
- `findings.json` — for a machine, schema below
- `queries/` — the scripts you actually ran, as `01-*.py`

`report.md` opens with a **Summary** of three to five lines that answers the question on
its own, and closes with **Data limits** — coverage gaps, unconfirmed matches, anything
a registry declined to say — and **Next steps**.
"
    );

    if let Some(p) = preset {
        s.push_str("\nBetween those, the body:\n\n");
        for (i, sec) in p.body_sections().iter().enumerate() {
            s.push_str(&format!("{}. {}\n", i + 1, sec));
        }
    } else {
        s.push_str("\nBetween those, a body organised to fit the question.\n");
    }

    s.push_str(
        "\n`findings.json` has a fixed schema, so two runs can be compared:\n\
         `{ \"run_id\", \"task\", \"entities\": [], \
         \"findings\": [{ \"severity\", \"statement\", \"evidence\": [], \"confidence\" }], \
         \"data_gaps\": [] }`\n",
    );

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> Paths<'static> {
        Paths {
            data: "./live",
            workspace: "./workspace",
            artifacts: "./artifacts",
        }
    }

    #[test]
    fn presets_round_trip() {
        for slug in Preset::ALL {
            let p = Preset::parse(slug).expect(slug);
            assert_eq!(p.slug(), slug);
            assert!(!p.task("Acme").is_empty());
        }
        assert!(Preset::parse("nope").is_none());
    }

    #[test]
    fn the_instruction_carries_policy_and_not_the_map() {
        let s = instruction(&paths(), Some(Preset::OwnershipTree), "run-1");
        assert!(s.contains("./live/CATALOG.md"));
        assert!(s.contains("./artifacts/run-1/"));
        assert!(s.contains("Read-only"));
        assert!(s.contains("possible match — needs confirmation"));
        assert!(s.contains("Write as you go"));

        // Path grammar, field names and per-registry quirks belong to the stores, which
        // serve them and can keep them true. Repeating any of it here would make the
        // instruction a second source that drifts.
        for leak in [
            "by-lei", "by-cik", "pages/", "ownedBy", "entity.legalAddress",
            "submissions.json", "cik_str", "facts.json",
        ] {
            assert!(!s.contains(leak), "instruction spells out the tree: {leak}");
        }
    }

    #[test]
    fn each_preset_asks_for_its_own_body() {
        let profile = instruction(&paths(), Some(Preset::EntityProfile), "run-1");
        let cross = instruction(&paths(), Some(Preset::CrossRegistry), "run-1");
        assert!(profile.contains("Registration and status"));
        assert!(cross.contains("Where they disagree"));
        assert!(!profile.contains("Where they disagree"));
        assert!(!cross.contains("Registration and status"));
    }
}
