//! `query` — read-only free-form SQL. Only when `search` and `read` cannot ask it.

use std::path::Path;

use super::Failure;

pub const USAGE: &str = "\
headhunting query — read-only SQL.

usage:
  headhunting query <sql> [--limit <n>]

For when something comes up that search and read cannot ask. Writes are refused by
SQLite itself. One statement at a time. The schema is in `schema.sql`.
";

/// How many rows an answer carries.
///
/// The same as `search`. Free-form SQL has no reason to be more generous, and if the two
/// differed the caller would have to remember which was which.
const DEFAULT_LIMIT: usize = 100;

pub fn run(db: &Path, args: &[&str]) -> Result<Vec<u8>, Failure> {
    let mut limit = DEFAULT_LIMIT;
    let mut statements: Vec<&str> = Vec::new();

    let mut it = args.iter().copied();
    while let Some(arg) = it.next() {
        match arg {
            "--limit" => {
                let value = it
                    .next()
                    .ok_or_else(|| Failure::usage("headhunting query: --limit takes a number"))?;
                limit = value.parse().map_err(|_| {
                    Failure::usage(format!(
                        "headhunting query: --limit: {value:?} is not a number"
                    ))
                })?;
            }
            // Everything after `--` is an argument, so SQL beginning with a dash is
            // still SQL.
            "--" => statements.extend(it.by_ref()),
            other if other.starts_with("--") => {
                return Err(Failure::usage(format!(
                    "headhunting query: unknown option {other:?}\n\n{USAGE}"
                )));
            }
            other => statements.push(other),
        }
    }

    let [sql] = statements.as_slice() else {
        return Err(Failure::usage(format!(
            "headhunting query: expected one statement, got {}\n\n{USAGE}",
            statements.len()
        )));
    };

    let conn = super::sql::open(db)?;
    // Several statements in one call are caught here. `prepare` readies the first and
    // silently drops the rest, which would make `SELECT 1; DELETE …` look like a success.
    let rows = super::sql::ask(&conn, sql, limit)
        .map_err(|e| Failure::failed(format!("headhunting query: {e}")))?;
    Ok(super::sql::table(&rows).into_bytes())
}

#[cfg(test)]
mod tests {
    use crate::executable::{Headhunting, tests::pool};
    use cortex::exec::{ExecCall, ExecResult, Executable};

    async fn run(db: &Path, args: &[&str]) -> ExecResult {
        let call = ExecCall {
            name: "headhunting".into(),
            args: std::iter::once("query")
                .chain(args.iter().copied())
                .map(str::to_string)
                .collect(),
            cwd: Some(String::new()),
            env: Default::default(),
        };
        Headhunting::new(db).exec(&call, None).await
    }

    use std::path::Path;

    async fn out(db: &Path, args: &[&str]) -> String {
        let result = run(db, args).await;
        assert_eq!(
            result.exit_code,
            0,
            "{args:?} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap()
    }

    #[tokio::test]
    async fn a_select_answers_with_a_header_and_the_rows() {
        let (_dir, db) = pool();
        let got = out(
            &db,
            &["SELECT name FROM candidate_brief WHERE city='Berlin'"],
        )
        .await;
        assert!(got.starts_with("name"), "the header comes first: {got:?}");
        assert!(got.contains("Blake Underhill"), "{got:?}");
    }

    #[tokio::test]
    async fn a_view_is_queried_like_a_table() {
        let (_dir, db) = pool();
        let got = out(
            &db,
            &["SELECT real_months FROM candidate_tenure WHERE id='urn:li:person:riley'"],
        )
        .await;
        assert!(got.contains("72"), "{got:?}");
    }

    /// Writes are not stopped by a parser here. The connection is read-only, so SQLite
    /// refuses them itself.
    #[tokio::test]
    async fn a_write_is_refused_and_the_data_is_untouched() {
        let (_dir, db) = pool();
        let result = run(&db, &["DELETE FROM candidates"]).await;
        assert_eq!(result.exit_code, 1);
        assert!(result.stdout.is_empty(), "a failure writes no data");

        let left: i64 = rusqlite::Connection::open(&db)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM candidates", [], |r| r.get(0))
            .unwrap();
        assert_eq!(left, 6, "not one person may be deleted");
    }

    /// **A parser would have missed this.** It begins with `WITH`, so reading only the
    /// first word calls it a read.
    #[tokio::test]
    async fn a_write_hiding_behind_a_cte_is_refused_too() {
        let (_dir, db) = pool();
        let result = run(
            &db,
            &["WITH x AS (SELECT 1 AS a) INSERT INTO skills SELECT 'urn:li:person:riley','Go',1 FROM x"],
        )
        .await;
        assert_eq!(result.exit_code, 1);
    }

    /// The one thing read-only does not stop. Attach and another database outside the
    /// mount becomes readable.
    #[tokio::test]
    async fn attaching_another_database_is_refused() {
        let (_dir, db) = pool();
        let result = run(
            &db,
            &[&format!("ATTACH DATABASE '{}' AS other", db.display())],
        )
        .await;
        assert_ne!(result.exit_code, 0, "ATTACH went through");
    }

    #[tokio::test]
    async fn a_second_statement_does_not_ride_along() {
        let (_dir, db) = pool();
        let result = run(&db, &["SELECT 1; DELETE FROM candidates"]).await;
        assert_ne!(result.exit_code, 0, "two statements went through");
    }

    #[tokio::test]
    async fn no_sql_is_a_usage_error() {
        let (_dir, db) = pool();
        let result = run(&db, &[]).await;
        assert_eq!(result.exit_code, 2);
    }

    #[tokio::test]
    async fn a_truncated_answer_says_how_many_there_were() {
        let (_dir, db) = pool();
        let got = out(&db, &["--limit", "2", "SELECT id FROM candidates"]).await;
        assert!(got.contains("-- 2 of 6 rows"), "{got:?}");
    }

    /// An answer that fits says nothing about counts. Saying it would make the reader
    /// check the arithmetic every time.
    #[tokio::test]
    async fn an_untruncated_answer_says_nothing_about_counts() {
        let (_dir, db) = pool();
        let got = out(&db, &["SELECT id FROM candidates"]).await;
        assert!(!got.contains(" rows"), "{got:?}");
    }
}
