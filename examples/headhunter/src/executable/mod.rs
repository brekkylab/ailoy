//! `headhunting` — the pool command this app attaches.
//!
//! # Why the app builds it
//!
//! cortex's `sqlite` is a command that stays ignorant of any schema on purpose. Its own
//! documentation gives the reason: a consumer that needs domain knowledge should put it
//! **in the database as views**, so that the command stays usable by the next consumer.
//! This module is the other side of that. It knows recruiting, and it has no use outside
//! this app.
//!
//! That is what the example is for. When an app attaches an executable in its own domain,
//! the agent asks in that domain's vocabulary rather than in SQL.
//!
//! # Three commands
//!
//! `search` answers with a table and `read` answers with a block. Both are `SELECT`s,
//! but the shape of the answer differs: `candidate_brief` is one row per person, while
//! `positions` and `skills` are several and do not fit on one line of a table.
//!
//! `query` is the back door. Sorting a real run's 29 queries by kind, not one of them
//! needed free-form SQL. It is kept because a question nobody anticipated will come up,
//! and with that shut there is nothing the agent can do.

use cortex::{
    exec::{ExecCall, ExecResult, Executable},
    fs::Mount,
};
use futures_core::future::BoxFuture;
use std::path::PathBuf;

mod distribution;
mod query;
mod read;
mod search;
mod sql;

/// What the command prints when asked.
pub const USAGE: &str = "\
headhunting — the candidate pool. A command this app attaches.

usage:
  headhunting search <conditions…>       people matching them, as a table (one line each)
  headhunting read <id…>                 the people you picked, in full (several lines each)
  headhunting distribution <axis> [term] what values an axis is made of
  headhunting query <sql>                read-only SQL. Only when the others cannot ask it.

For detail: `headhunting <command> --help`.
The schema is in `schema.sql`.
";

/// The `headhunting` command.
///
/// # Why it holds a database path
///
/// cortex's `sqlite` holds nothing: every call names its database, and what is inside is
/// not that command's business. This is the opposite. There is one pool and the app knows
/// where it is, so the command line carries no db argument.
pub struct Headhunting {
    db: PathBuf,
}

impl Headhunting {
    pub fn new(db: impl Into<PathBuf>) -> Headhunting {
        Headhunting { db: db.into() }
    }

    /// One line, for `ExecutableSet::register`.
    pub fn summary() -> &'static str {
        "search, read, and query the candidate pool"
    }

    async fn run(&self, call: &ExecCall) -> Result<Vec<u8>, Failure> {
        let mut args = call.args.iter().map(String::as_str);
        let Some(command) = args.next() else {
            return Err(Failure::usage(format!(
                "headhunting: no command\n\n{USAGE}"
            )));
        };
        let rest: Vec<&str> = args.collect();

        // A subcommand's `--help` is not left to that command's argument parser. If the
        // parser checked arguments first, `search --help` could end in "no conditions",
        // which answers nothing for someone who asked how to use it.
        let wants_help = rest.iter().any(|a| *a == "-h" || *a == "--help");

        match command {
            "-h" | "--help" => Ok(USAGE.into()),
            "search" if wants_help => Ok(search::USAGE.into()),
            "distribution" if wants_help => Ok(distribution::USAGE.into()),
            "read" if wants_help => Ok(read::USAGE.into()),
            "query" if wants_help => Ok(query::USAGE.into()),
            "search" => search::run(&self.db, &rest),
            "distribution" => distribution::run(&self.db, &rest),
            "read" => read::run(&self.db, &rest),
            "query" => query::run(&self.db, &rest),
            other => Err(Failure::usage(format!(
                "headhunting: unknown command {other:?}\n\n{USAGE}"
            ))),
        }
    }
}

/// A command that did not happen, and what to exit with.
///
/// Two codes, because a caller can act on the difference: `2` is a command that was not
/// understood and can be reissued differently, `1` is one that was understood and did not
/// work. cortex's `sqlite` draws the same line.
#[derive(Debug)]
pub(crate) struct Failure {
    pub code: i32,
    pub message: String,
}

impl Failure {
    pub fn usage(message: impl Into<String>) -> Failure {
        Failure {
            code: 2,
            message: message.into(),
        }
    }

    pub fn failed(message: impl Into<String>) -> Failure {
        Failure {
            code: 1,
            message: message.into(),
        }
    }
}

impl Executable for Headhunting {
    fn exec<'a>(
        &'a self,
        call: &'a ExecCall,
        _mount: Option<&'a dyn Mount>,
    ) -> BoxFuture<'a, ExecResult> {
        Box::pin(async move {
            match self.run(call).await {
                Ok(stdout) => ExecResult::ok(stdout),
                Err(f) => ExecResult::failed(f.code, format!("{}\n", f.message.trim_end())),
            }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A temporary database built from the real schema and views.
    ///
    /// The files are baked in at compile time. Reading them by relative path would make
    /// the test depend on where it runs, and copying the schema into the test would let it
    /// keep passing after the real schema changed.
    const SCHEMA: &str = include_str!("../../sql/schema.sql");
    const VIEWS: &str = include_str!("../../sql/views.sql");

    /// The SQL that fills FTS5. Must match `sql/load.py`.
    ///
    /// A candidate's several rows are concatenated into one document, because
    /// `MATCH 'rust'` has to find that person whether it sits in a skill or a headline.
    const FILL_FTS: &str = "
        INSERT INTO candidate_fts (id, headline, summary, titles, descriptions, skill_names)
        SELECT c.id, c.headline, c.summary,
               (SELECT group_concat(title, ' ') FROM positions WHERE candidate_id = c.id),
               (SELECT group_concat(description, ' ') FROM positions WHERE candidate_id = c.id),
               (SELECT group_concat(name, ' ') FROM skills WHERE candidate_id = c.id)
        FROM candidates c";

    /// A pool of six people.
    ///
    /// The real dataset's traps, scaled down. Not just six people — **each one is here to
    /// test something specific**, so the tests check whether the command handles the
    /// domain rather than whether it merely runs.
    ///
    /// | id | what it tests |
    /// | --- | --- |
    /// | riley | two spans overlap by 24 months. Summed 96, merged 72 |
    /// | casey | Rust is in the headline only, not the skill list. And no way to reach them |
    /// | jordan | Rust is in the skill list with no evidence in any position description |
    /// | rowan | passes every condition but `open_to_work` is 0 |
    /// | jihun | the only skill is `러스트`, so `MATCH 'rust'` misses them. Lives in Seongnam |
    /// | blake | lives in Berlin. Checks that the city condition actually filters |
    pub(crate) fn pool() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("pool.db");
        let conn = rusqlite::Connection::open(&path).expect("a new database");
        conn.execute_batch(SCHEMA).expect("the schema");
        conn.execute_batch(VIEWS).expect("the views");

        // (id, first, last, headline, summary, city, language, open_to_work)
        let people = [
            (
                "riley",
                "Riley",
                "Calloway",
                "Backend Engineer · Rust, distributed systems",
                "Rewrote the settlement batch in Rust.",
                "Seoul",
                "en",
                1,
            ),
            (
                "casey",
                "Casey",
                "Ashby",
                "Backend Engineer · Python · learning Rust",
                "Reading about it on my own time; nothing shipped yet.",
                "Seoul",
                "en",
                1,
            ),
            (
                "jordan",
                "Jordan",
                "Merrick",
                "Senior Backend Engineer · Java, Kafka",
                "Ran the payments pipeline.",
                "Seoul",
                "en",
                1,
            ),
            (
                "rowan",
                "Rowan",
                "Voss",
                "Staff Backend Engineer · Rust",
                "Designed idempotency for distributed systems.",
                "Seoul",
                "en",
                0,
            ),
            (
                "jihun",
                "지훈",
                "도",
                "백엔드 엔지니어 · 서버 개발",
                "대용량 트래픽 처리를 담당했다.",
                "Seongnam",
                "ko",
                1,
            ),
            (
                "blake",
                "Blake",
                "Underhill",
                "Backend Engineer · Rust",
                "Used Rust on the embedded side.",
                "Berlin",
                "en",
                1,
            ),
        ];
        for (id, first, last, headline, summary, city, lang, open) in people {
            conn.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,'Software','Engineering',
                 'Senior',?,?,300,'2026-06-01','https://example.test')",
                rusqlite::params![
                    format!("urn:li:person:{id}"),
                    first,
                    last,
                    headline,
                    summary,
                    city,
                    if city == "Berlin" { "DE" } else { "KR" },
                    lang,
                    open
                ],
            )
            .expect("a candidate");
        }

        // (id, ord, title, company, company urn, description, start year, start month,
        //  end year, end month)
        //
        // casey's "Pinehurst" and jordan's "Pinehurst Systems" are the same company under
        // two spellings, and only the urn says so. That is how a duplicate profile hides.
        //
        // riley's two rows overlap from 2022-01 to 2024-01. Summed that is 96 months;
        // merged it runs 2020-01 to 2026-01, which is 72.
        let jobs = [
            (
                "riley",
                0,
                "Backend Engineer",
                "Sentinel Freight",
                "urn:li:company:11",
                "Rebuilt distributed job handling in Rust.",
                2020,
                1,
                Some(2024),
                Some(1),
            ),
            (
                "riley",
                1,
                "Backend Engineer",
                "Kestrel Labs",
                "urn:li:company:12",
                "Designed idempotency for the settlement batch.",
                2022,
                1,
                Some(2026),
                Some(1),
            ),
            (
                "casey",
                0,
                "Backend Engineer",
                "Pinehurst",
                "urn:li:company:13",
                "Built APIs with Python and PostgreSQL.",
                2017,
                1,
                None,
                None,
            ),
            (
                "jordan",
                0,
                "Senior Backend Engineer",
                "Pinehurst Systems",
                "urn:li:company:13",
                "Ran the payments pipeline on Java and Kafka.",
                2018,
                1,
                None,
                None,
            ),
            (
                "rowan",
                0,
                "Staff Backend Engineer",
                "Halberd",
                "urn:li:company:14",
                "Designed retries and failover in Rust.",
                2018,
                1,
                None,
                None,
            ),
            (
                "jihun",
                0,
                "백엔드 엔지니어",
                "누리테크",
                "urn:li:company:15",
                "러스트로 대용량 트래픽 처리를 맡았다.",
                2018,
                1,
                None,
                None,
            ),
            (
                "blake",
                0,
                "Backend Engineer",
                "Osterhagen",
                "urn:li:company:16",
                "Built the upper firmware layer in Rust.",
                2021,
                1,
                None,
                None,
            ),
        ];
        for (id, ord, title, company, company_urn, description, sy, sm, ey, em) in jobs {
            // riley's second role is a contract. An overlap has to be visible as an
            // employment type and not only as a month count, or a reader cannot tell
            // concurrent employment from a gap in the record.
            let employment = if id == "riley" && ord == 1 {
                "CONTRACT"
            } else {
                "FULL_TIME"
            };
            conn.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,'201-500',
                 ?,'HYBRID','Seoul',?,?,?,?,?)",
                rusqlite::params![
                    format!("urn:li:person:{id}"),
                    ord,
                    title,
                    company,
                    company_urn,
                    employment,
                    description,
                    sy,
                    sm,
                    ey,
                    em
                ],
            )
            .expect("a position");
        }

        // casey has Rust in the headline only, so not in the skill list. jihun has only
        // `러스트`, which `MATCH 'rust'` does not reach.
        let skills = [
            ("riley", "Rust", 31),
            ("riley", "PostgreSQL", 24),
            ("casey", "Python", 40),
            ("jordan", "Rust", 12),
            ("jordan", "Kafka", 30),
            ("rowan", "Rust", 20),
            ("jihun", "러스트", 8),
            ("blake", "rust-lang", 6),
        ];
        for (id, name, endorsements) in skills {
            conn.execute(
                "INSERT INTO skills VALUES (?,?,?)",
                rusqlite::params![format!("urn:li:person:{id}"), name, endorsements],
            )
            .expect("a skill");
        }

        // casey has no row. That is what it means to be unreachable.
        for id in ["riley", "jordan", "rowan", "jihun", "blake"] {
            conn.execute(
                "INSERT INTO contacts VALUES (?,'inmail','reachable by InMail')",
                rusqlite::params![format!("urn:li:person:{id}")],
            )
            .expect("a contact");
        }

        // The four tables `read` used to leave out. Each mirrors a trap in the real pool:
        // a degree without the practice behind it, a certificate without it, a fluency
        // entry that tempts you away from `profile_language`, and a desired arrangement
        // that contradicts `open_to_work` being true.
        conn.execute(
            "INSERT INTO educations VALUES ('urn:li:person:jordan','Hanyang University',
             'Master of Science','Computer Science',2014,2016)",
            [],
        )
        .expect("an education");
        conn.execute(
            "INSERT INTO certifications VALUES ('urn:li:person:rowan',
             'Certified Kubernetes Administrator','Cloud Native Computing Foundation')",
            [],
        )
        .expect("a certification");
        conn.execute(
            "INSERT INTO languages VALUES ('urn:li:person:jihun','English','NATIVE_OR_BILINGUAL')",
            [],
        )
        .expect("a language");
        conn.execute(
            "INSERT INTO open_to_work_prefs VALUES ('urn:li:person:riley','Backend Engineer',
             'Remote','Seoul','2026-09','FULL_TIME')",
            [],
        )
        .expect("a preference");

        conn.execute_batch(FILL_FTS).expect("the FTS index");
        drop(conn);
        (dir, path)
    }

    fn call(args: &[&str]) -> ExecCall {
        ExecCall {
            name: "headhunting".into(),
            args: args.iter().map(|a| a.to_string()).collect(),
            cwd: Some(String::new()),
            env: Default::default(),
        }
    }

    /// What the command answered on stdout. On failure it panics with stderr.
    async fn out(db: &std::path::Path, args: &[&str]) -> String {
        let result = Headhunting::new(db).exec(&call(args), None).await;
        assert_eq!(
            result.exit_code,
            0,
            "{args:?} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).expect("it writes text")
    }

    #[tokio::test]
    async fn help_is_the_usage_on_stdout() {
        let result = Headhunting::new("nowhere.db")
            .exec(&call(&["--help"]), None)
            .await;
        assert_eq!(result.exit_code, 0);
        let out = String::from_utf8(result.stdout).expect("it writes text");
        assert!(out.starts_with("headhunting —"), "{out:?}");
    }

    #[tokio::test]
    async fn each_command_answers_its_own_help() {
        let db = std::path::Path::new("nowhere.db");
        for (cmd, head) in [
            ("search", "headhunting search —"),
            ("read", "headhunting read —"),
            ("query", "headhunting query —"),
        ] {
            let got = out(db, &[cmd, "--help"]).await;
            assert!(got.starts_with(head), "{cmd}: {got:?}");
        }
    }

    #[tokio::test]
    async fn an_unknown_command_is_a_usage_error_and_says_what_there_is() {
        let result = Headhunting::new("nowhere.db")
            .exec(&call(&["find"]), None)
            .await;
        // 2 rather than 1: the command was not understood and can be reissued differently.
        assert_eq!(result.exit_code, 2);
        let err = String::from_utf8_lossy(&result.stderr);
        assert!(
            err.contains("find"),
            "it has to name what it did not know: {err:?}"
        );
        assert!(err.contains("search"), "and what there is: {err:?}");
    }

    #[tokio::test]
    async fn no_command_at_all_is_a_usage_error() {
        let result = Headhunting::new("nowhere.db").exec(&call(&[]), None).await;
        assert_eq!(result.exit_code, 2);
        assert!(result.stdout.is_empty(), "a failure writes no data");
    }

    /// Runs each of the three commands once against the real pool.
    ///
    /// The fixture has six people; the real data has 600. The risk is not scale but
    /// **shape** — a view may produce values the fixture never does, or the FTS documents
    /// may be filled differently.
    ///
    /// `#[ignore]` because the data may not be in the repository. To run it:
    /// `cargo test -p headhunter -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "needs the real data/headhunter.db"]
    async fn the_commands_work_against_the_real_pool() {
        let db = std::path::Path::new("data/headhunter.db");
        assert!(db.is_file(), "no pool. Build it with `python3 sql/load.py`");

        let found = out(
            db,
            &[
                "search",
                "--skill",
                "rust",
                "--city",
                "Seoul,Seongnam",
                "--min-years",
                "4",
            ],
        )
        .await;
        println!("--- search ---\n{found}");
        assert!(found.contains("urn:li:person:"), "{found}");

        let id = found
            .lines()
            .find_map(|l| {
                l.starts_with("urn:li:person:")
                    .then(|| l.split_whitespace().next())
            })
            .flatten()
            .expect("one id");
        let profile = out(db, &["read", id]).await;
        println!("--- read ---\n{profile}");
        assert!(profile.contains("tenure"), "{profile}");

        let counted = out(db, &["query", "SELECT COUNT(*) AS n FROM candidates"]).await;
        println!("--- query ---\n{counted}");
        assert!(counted.starts_with('n'), "{counted}");
    }

    /// Whether the fixture actually carries what it claims to test.
    ///
    /// If this is wrong, every test below checks bad data rather than the command.
    #[test]
    fn the_pool_carries_the_traps_it_claims_to() {
        let (_dir, db) = pool();
        let conn = rusqlite::Connection::open(&db).unwrap();

        let (naive, real): (i64, i64) = conn
            .query_row(
                "SELECT naive_months, real_months FROM candidate_tenure
                 WHERE id='urn:li:person:riley'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            (naive, real),
            (96, 72),
            "riley's two spans overlap by 24 months"
        );

        let contacts: i64 = conn
            .query_row(
                "SELECT contact_rows FROM candidate_brief WHERE id='urn:li:person:casey'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(contacts, 0, "casey has no way to be reached");

        let found: Vec<String> = conn
            .prepare("SELECT id FROM candidate_fts WHERE candidate_fts MATCH 'rust'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert!(
            !found.iter().any(|id| id.ends_with("jihun")),
            "`러스트` shares no token with `rust`: {found:?}"
        );
        assert!(
            found.iter().any(|id| id.ends_with("casey")),
            "casey is in the index even with it only in the headline: {found:?}"
        );
    }
}
