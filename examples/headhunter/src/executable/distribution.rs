//! `distribution` — what the pool holds along one axis, and how many carry each value.
//!
//! # Why this is its own command
//!
//! Two questions need it, and only the first was foreseen.
//!
//! **What values does a condition accept?** `--city` matches exactly, so passing a name
//! that is not in the pool returns nobody — indistinguishable from nobody fitting. And
//! `--skill` goes through a full-text index, so a spelling that shares no characters with
//! your term is simply not there.
//!
//! **What kind of people are in here at all?** A posting whose right answer is "nobody"
//! cannot be answered by searching, because an empty result means either that the
//! vocabulary is absent or that it is spelled otherwise. Telling those apart means looking
//! at the titles and companies the pool actually holds. No condition asks that question,
//! which is why the axes are not limited to the ones a condition feeds.
//!
//! This used to be a `--spellings` flag on `search` that did no searching, covered skills
//! only, and took no argument — its help had to say "instead of searching", and the moment
//! it was set the other eight conditions became meaningless. A real run called
//! `--spellings rust` and got a usage error for it.
//!
//! Looking at values is not searching for people. It gets its own command.
//!
//! # `city` does not come from `location_distribution`
//!
//! That view reports `positions.location`, whose values drift — `Seoul, KR`,
//! `Greater Seoul Area`, and `Seoul, South Korea` are the same place. `--city` matches the
//! normalized `candidates.city`. Reading the view and passing its value to `--city`
//! returns nobody, and two separate runs lost turns to exactly that. So `distribution
//! city` counts `candidates.city` — the values `--city` actually accepts.

use std::path::Path;

use super::Failure;

pub const USAGE: &str = "\
headhunting distribution — what values a condition accepts, and how many
carry each.

usage:
  headhunting distribution <axis> [term]

axes:
  skill          skill names, as written on profiles
  city           the cities people work in — the values --city accepts
  language       the profile languages — the values --language accepts
  title          the job titles held in the pool
  company        the companies people work or worked at
  certification  the certificates people hold

Give a term and only values containing it are listed; leave it out for all of them.

Spelling drifts. The same skill is written under several names, and some of them share no
characters with the others, so a term will not list them. Run the axis without a term to
see everything it holds.
";

/// How many values an answer carries.
///
/// Larger than `search`'s: the point of this command is seeing the whole shape, and these
/// are one short value per line rather than a person per line.
const DEFAULT_LIMIT: usize = 200;

/// Each axis, and where its values are counted from.
///
/// **An axis is here because someone had to reach past this command to see it.** The four
/// runs of run-7 went to free-form SQL for `title` three times, `company_name` four times,
/// and the certificate list once. Counted that way `company` is the most-wanted axis in
/// the whole tool.
///
/// That reasoning replaces the one this file used to carry — "an axis is here because a
/// condition accepts its values" — which kept `title` out on the grounds that there is no
/// `--title` to feed. The posting whose right answer is "nobody" broke it: proving a
/// vocabulary is absent rather than differently spelled means first knowing what kind of
/// people the pool holds, and no condition asks that question.
///
/// `job_function` and `industry` were each reached for once and are left out for now. That
/// is the same judgment that was wrong about `title`, so the next run may overturn it.
///
/// `city` deliberately does not use `location_distribution`. See the module note.
const AXES: &[Axis] = &[
    Axis {
        name: "skill",
        value: "name",
        from: "skills",
        per: PEOPLE,
    },
    Axis {
        name: "city",
        value: "city",
        from: "candidates",
        per: ROWS,
    },
    Axis {
        name: "language",
        value: "profile_language",
        from: "candidates",
        per: ROWS,
    },
    Axis {
        name: "title",
        value: "title",
        from: "positions",
        per: PEOPLE,
    },
    Axis {
        name: "company",
        value: "company_name",
        from: "positions",
        per: PEOPLE,
    },
    Axis {
        name: "certification",
        value: "name",
        from: "certifications",
        per: PEOPLE,
    },
];

struct Axis {
    name: &'static str,
    /// The column holding the value. **Not always called `name`**, which is why the SQL is
    /// built from this rather than written out per axis — the earlier version wrote one
    /// query per axis and then string-replaced the column back into the `WHERE` clause.
    value: &'static str,
    from: &'static str,
    per: &'static str,
}

/// One row per person, so someone with a skill listed twice counts once.
const PEOPLE: &str = "COUNT(DISTINCT candidate_id)";
/// `candidates` already holds one row per person.
const ROWS: &str = "COUNT(*)";

impl Axis {
    /// `(sql, binds)`. With a term the value is matched as a substring rather than through
    /// the full-text index: this lists values, and a value is short; the index is for
    /// finding people through prose.
    fn query(&self, term: Option<&str>) -> (String, Vec<String>) {
        let Axis {
            value, from, per, ..
        } = self;
        match term {
            Some(t) => (
                format!(
                    "SELECT {value} AS name, {per} AS people FROM {from}
                     WHERE lower({value}) LIKE '%' || lower(?) || '%'
                     GROUP BY {value} ORDER BY 2 DESC, 1"
                ),
                vec![t.to_string()],
            ),
            None => (
                format!(
                    "SELECT {value} AS name, {per} AS people FROM {from}
                     GROUP BY {value} ORDER BY 2 DESC, 1"
                ),
                Vec::new(),
            ),
        }
    }

    fn total(&self) -> String {
        let Axis { value, from, .. } = self;
        format!("SELECT COUNT(DISTINCT {value}) FROM {from}")
    }
}

pub fn run(db: &Path, args: &[&str]) -> Result<Vec<u8>, Failure> {
    let mut limit = DEFAULT_LIMIT;
    let mut positional: Vec<&str> = Vec::new();

    let mut it = args.iter().copied();
    while let Some(arg) = it.next() {
        match arg {
            "--limit" => {
                let raw = it.next().ok_or_else(|| {
                    Failure::usage("headhunting distribution: --limit takes a number")
                })?;
                limit = raw.parse().map_err(|_| {
                    Failure::usage(format!(
                        "headhunting distribution: --limit: {raw:?} is not a number"
                    ))
                })?;
            }
            other if other.starts_with("--") => {
                return Err(Failure::usage(format!(
                    "headhunting distribution: unknown option {other:?}\n\n{USAGE}"
                )));
            }
            other => positional.push(other),
        }
    }

    let (axis, term) = match positional.as_slice() {
        [axis] => (*axis, None),
        [axis, term] => (*axis, Some(*term)),
        _ => {
            let names: Vec<&str> = AXES.iter().map(|a| a.name).collect();
            return Err(Failure::usage(format!(
                "headhunting distribution: takes an axis and an optional term. Axes: {}\n\n{USAGE}",
                names.join(" ")
            )));
        }
    };

    let Some(axis_def) = AXES.iter().find(|a| a.name == axis) else {
        let names: Vec<&str> = AXES.iter().map(|a| a.name).collect();
        return Err(Failure::usage(format!(
            "headhunting distribution: unknown axis {axis:?}. Available: {}",
            names.join(" ")
        )));
    };

    let (sql, binds) = axis_def.query(term);

    let conn = super::sql::open(db)?;
    let rows = super::sql::ask_with(&conn, &sql, &binds, limit)
        .map_err(|e| Failure::failed(format!("headhunting distribution: {e}")))?;

    let mut out = String::new();
    if rows.rows.is_empty() {
        out.push_str(&match term {
            Some(t) => format!("no {axis} value contains {t:?}\n"),
            None => format!("the {axis} axis is empty\n"),
        });
        return Ok(out.into_bytes());
    }
    // Say how many the axis holds in total, so a filtered list does not read as the whole.
    if term.is_some() {
        let total: i64 = conn
            .query_row(&axis_def.total(), [], |r| r.get(0))
            .map_err(|e| Failure::failed(format!("headhunting distribution: {e}")))?;
        out.push_str(&format!(
            "{} of the {total} values on the {axis} axis\n\n",
            rows.total
        ));
    }
    out.push_str(&super::sql::table(&rows));
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::{Headhunting, tests::pool};
    use cortex::exec::{ExecCall, ExecResult, Executable};

    async fn run_it(db: &Path, args: &[&str]) -> ExecResult {
        let call = ExecCall {
            name: "headhunting".into(),
            args: std::iter::once("distribution")
                .chain(args.iter().copied())
                .map(str::to_string)
                .collect(),
            cwd: Some(String::new()),
            env: Default::default(),
        };
        Headhunting::new(db).exec(&call, None).await
    }

    async fn out(db: &Path, args: &[&str]) -> String {
        let result = run_it(db, args).await;
        assert_eq!(
            result.exit_code,
            0,
            "{args:?} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap()
    }

    /// **The axis a real run had to reach through free-form SQL.**
    ///
    /// `--city` matches `candidates.city`, and there was no way to see those values.
    /// `location_distribution` reports `positions.location`, which drifts, so reading it
    /// and passing the value returns nobody. Two runs lost turns to that.
    #[tokio::test]
    async fn city_lists_the_values_that_city_accepts() {
        let (_dir, db) = pool();
        let got = out(&db, &["city"]).await;
        assert!(got.contains("Seoul"), "{got}");
        assert!(got.contains("Seongnam"), "{got}");
        assert!(got.contains("Berlin"), "{got}");
    }

    /// The skill axis is what `--spellings` used to be.
    #[tokio::test]
    async fn skill_lists_every_spelling_including_the_unreachable_ones() {
        let (_dir, db) = pool();
        let got = out(&db, &["skill"]).await;
        assert!(got.contains("Rust"), "{got}");
        assert!(
            got.contains("러스트"),
            "the one no rust term reaches: {got}"
        );
    }

    /// A term narrows the list — and the agent tried exactly this before it existed.
    #[tokio::test]
    async fn a_term_narrows_the_list() {
        let (_dir, db) = pool();
        let got = out(&db, &["skill", "rust"]).await;
        assert!(got.contains("Rust"), "{got}");
        assert!(
            !got.contains("PostgreSQL"),
            "an unrelated value survived: {got}"
        );
    }

    /// A filtered list that does not say what it was filtered out of reads as the whole.
    #[tokio::test]
    async fn a_filtered_list_says_how_many_the_axis_holds() {
        let (_dir, db) = pool();
        let got = out(&db, &["skill", "rust"]).await;
        let head = got.lines().next().unwrap();
        assert!(head.contains("of the"), "{head:?}");
    }

    /// The pool holds three profile languages, and one posting gates on them.
    #[tokio::test]
    async fn language_is_an_axis_too() {
        let (_dir, db) = pool();
        let got = out(&db, &["language"]).await;
        assert!(got.contains("ko"), "{got}");
        assert!(got.contains("en"), "{got}");
    }

    /// **The axis I took out as unused, put back by the run that used it.**
    ///
    /// `distribution.rs` argued that `title` was consulted once across five runs and never
    /// again, and that with no `--title` condition its values had nowhere to go. Then a
    /// posting came along whose right answer was "nobody", and proving that requires
    /// knowing what kind of people the pool holds at all. Three runs reached for
    /// `select distinct title from positions` through free-form SQL.
    #[tokio::test]
    async fn title_is_an_axis() {
        let (_dir, db) = pool();
        let got = out(&db, &["title"]).await;
        assert!(got.contains("Backend Engineer"), "{got}");
    }

    /// The most-reached-for axis of all: four calls across the four runs.
    #[tokio::test]
    async fn company_is_an_axis() {
        let (_dir, db) = pool();
        let got = out(&db, &["company"]).await;
        assert!(got.contains("Pinehurst"), "{got}");
        assert!(got.contains("Sentinel Freight"), "{got}");
    }

    /// `read` shows a person's certificates; this is the only way to see what the pool
    /// holds.
    #[tokio::test]
    async fn certification_is_an_axis() {
        let (_dir, db) = pool();
        let got = out(&db, &["certification"]).await;
        assert!(got.contains("Certified Kubernetes Administrator"), "{got}");
    }

    /// A term has to narrow every axis, not only the one whose value column happens to be
    /// called `name`.
    #[tokio::test]
    async fn a_term_narrows_the_axes_whose_column_is_not_called_name() {
        let (_dir, db) = pool();
        let got = out(&db, &["city", "seo"]).await;
        assert!(got.contains("Seoul"), "{got}");
        assert!(got.contains("Seongnam"), "{got}");
        assert!(!got.contains("Berlin"), "an unrelated city survived: {got}");
    }

    #[tokio::test]
    async fn an_unknown_axis_says_what_there_is() {
        let (_dir, db) = pool();
        // This used to name `company`, which has since become an axis. What the test is
        // about is the message, not the word.
        let result = run_it(&db, &["seniority"]).await;
        assert_eq!(result.exit_code, 2);
        let err = String::from_utf8_lossy(&result.stderr);
        assert!(err.contains("seniority"), "{err}");
        assert!(
            err.contains("skill"),
            "it has to say what is available: {err}"
        );
    }

    #[tokio::test]
    async fn no_axis_at_all_is_a_usage_error() {
        let (_dir, db) = pool();
        assert_eq!(run_it(&db, &[]).await.exit_code, 2);
    }

    /// A term that matches nothing says so rather than answering with an empty table.
    #[tokio::test]
    async fn a_term_that_matches_nothing_says_so() {
        let (_dir, db) = pool();
        let got = out(&db, &["skill", "cobol"]).await;
        assert!(got.contains("no skill value contains"), "{got}");
    }
}
