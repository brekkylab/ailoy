//! `search` — build a candidate set from conditions. One line per person.

use std::path::Path;

use super::Failure;

pub const USAGE: &str = "\
headhunting search — build a candidate set from conditions. One line per person.

usage:
  headhunting search [conditions…] [--columns <list>] [--limit <n>]

conditions (give several and all must hold):
  --skill <word>     looks only at the skill list. **Repeatable** — give it three times
                     and only people carrying all three come back
  --mentions <word>  looks anywhere in the profile (headline, summary, titles,
                     position descriptions, skills). Repeatable, same meaning
  --city <list>      work city. Comma-separated, and repeating it widens the list. Must
                     match exactly; give a name that is not there and it tells you which
                     cities the pool holds
  --min-years <n>    real experience. Concurrent employment counts once, not twice
  --name <name>      by name. If several people share it, all of them come back
  --language <code>  the profile's language. `distribution language` for the values
  --id <list>        search only within these people. Comma-separated, repeatable

display:
  --columns <list>   columns to show. Default name,years,open,contact,headline
                     also available: city country seniority job_function language
                     naive_years title company company_size updated
  --limit <n>        default 100. When truncated the last line says how many of how many

The first line of an answer says which spellings this search caught and how many the pool
holds. **Spelling drifts.** The same skill is written under several names, and some of
them share no characters with your search term, so they are not found. If the count looks
short, run `headhunting distribution skill` to see all of them and widen.

`years` has concurrent employment merged; `naive_years` is what summing the spans gives.
When they differ, the spans overlap.

The id is always shown. 283 of the 600 share a name with someone else, so a name does not
identify a person here.
";

const DEFAULT_LIMIT: usize = 100;

/// The columns shown by default.
///
/// That `years`, `open`, and `contact` are here is this command's policy. They are
/// material for the judgment, and they are not accepted as conditions: filter them away
/// in the tool and the agent never learns that a strong candidate who isn't looking, or
/// one who cannot be reached, existed at all — and cannot say in the shortlist why they
/// were passed over.
const DEFAULT_COLUMNS: &[&str] = &["name", "years", "open", "contact", "headline"];

/// The selectable columns and the expression behind each.
///
/// These are `candidate_brief`'s columns. A few are renamed to save width in the table
/// (`real_years` → `years`); the meaning is the same.
const COLUMNS: &[(&str, &str)] = &[
    ("name", "b.name"),
    ("headline", "b.headline"),
    ("city", "b.city"),
    ("country", "b.country"),
    ("seniority", "b.seniority"),
    ("job_function", "b.job_function"),
    ("language", "b.profile_language"),
    ("open", "CASE b.open_to_work WHEN 1 THEN 'y' ELSE 'n' END"),
    // The method, not a count. Writing a mail needs to know *how* to reach them, and a
    // count sent the agent back through `read` — or straight to SQL — for every shortlist.
    // An empty cell still means zero, so nothing is lost.
    (
        "contact",
        "COALESCE((SELECT group_concat(method) FROM contacts x WHERE x.candidate_id = b.id), '')",
    ),
    ("years", "b.real_years"),
    // What summing the spans gives. Beside `years` it shows an overlap at a glance —
    // three runs went to free-form SQL for exactly this pair.
    (
        "naive_years",
        "(SELECT ROUND(t.naive_months / 12.0, 1) FROM candidate_tenure t WHERE t.id = b.id)",
    ),
    ("title", "b.current_title"),
    ("company", "b.current_company"),
    ("company_size", "b.current_company_size"),
    ("updated", "b.last_updated_at"),
];

pub fn run(db: &Path, args: &[&str]) -> Result<Vec<u8>, Failure> {
    let args = Args::parse(args)?;
    let conn = super::sql::open(db)?;

    let mut where_parts: Vec<String> = Vec::new();
    let mut binds: Vec<String> = Vec::new();

    // `--skill` restricts FTS5 to the `skill_names` column. `--mentions` names no column,
    // so it also reads headline, summary, titles, and position descriptions. That
    // difference separates someone who only wrote the skill in their headline from
    // someone who actually did the work.
    // One clause per term. `where_parts` is joined with AND, so several terms mean
    // everyone who satisfies all of them.
    for skill in &args.skills {
        where_parts
            .push("b.id IN (SELECT id FROM candidate_fts WHERE candidate_fts MATCH ?)".into());
        binds.push(format!("skill_names:{}", fts_term(skill)));
    }
    for mentions in &args.mentions {
        where_parts
            .push("b.id IN (SELECT id FROM candidate_fts WHERE candidate_fts MATCH ?)".into());
        binds.push(fts_term(mentions));
    }
    if !args.cities.is_empty() {
        check_cities(&conn, &args.cities)?;
        where_parts.push(format!("b.city IN ({})", holes(args.cities.len())));
        binds.extend(args.cities.iter().cloned());
    }
    if let Some(years) = args.min_years {
        // `CAST` because the value arrives bound as text. The view's `real_years` is a
        // computed result with no column affinity, so without it the comparison would be
        // lexical rather than numeric.
        where_parts.push("b.real_years >= CAST(? AS REAL)".into());
        binds.push(years.to_string());
    }
    if let Some(name) = &args.name {
        where_parts.push("b.name = ?".into());
        binds.push(name.clone());
    }
    // **This was added for a must-have that turned out not to be one.** A posting's gate
    // named `profile_language`, and it is gone now: the posting never asked for it, and
    // filtering on it let an agent pass the language test by writing Korean to everyone.
    //
    // The condition stays because the pool holds three profile languages and asking for
    // one of them is a fair question — reaching the 38 `ja` profiles, say. What it must
    // not be used for is what it was added for.
    if let Some(language) = &args.language {
        where_parts.push("b.profile_language = ?".into());
        binds.push(language.clone());
    }
    if !args.ids.is_empty() {
        where_parts.push(format!("b.id IN ({})", holes(args.ids.len())));
        binds.extend(args.ids.iter().cloned());
    }

    let selected: Vec<String> = args
        .columns
        .iter()
        .map(|name| {
            let expr = COLUMNS
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, e)| *e)
                .expect("the parser already checked this");
            format!("{expr} AS \"{name}\"")
        })
        .collect();

    // The id is not selectable; it is always the first column. A name does not identify a
    // person in this pool — 283 of the 600 share one with someone else.
    let sql = format!(
        "SELECT b.id AS \"id\", {} FROM candidate_brief b WHERE {} ORDER BY b.real_years DESC, b.id",
        selected.join(", "),
        where_parts.join(" AND ")
    );

    let rows = super::sql::ask_with(&conn, &sql, &binds, args.limit)
        .map_err(|e| Failure::failed(format!("headhunting search: {e}")))?;

    let mut out = String::new();
    // A line per term. Report only the first and a search that caught nothing on one of
    // the others reads as though it caught everything.
    for skill in &args.skills {
        out.push_str(&caught(&conn, skill, true)?);
    }
    for mentions in &args.mentions {
        out.push_str(&caught(&conn, mentions, false)?);
    }
    out.push_str(&super::sql::table(&rows));
    Ok(out.into_bytes())
}

/// Which spellings this search caught, and how many the pool holds in total.
///
/// **Spellings that were not caught are not found for you.** The same skill is also
/// written under names that share no characters at all (`Rust` and `러스트`), so the
/// relation is not visible in the string; a dictionary that knew it would be widening the
/// search rather than reporting it. What is reported instead is that spelling drifts, how
/// many there are in total, and how to see all of them.
fn caught(conn: &rusqlite::Connection, term: &str, skill_only: bool) -> Result<String, Failure> {
    let fail = |e: rusqlite::Error| Failure::failed(format!("headhunting search: {e}"));

    let mut stmt = conn
        .prepare(
            "SELECT name, COUNT(DISTINCT candidate_id) FROM skills
             WHERE lower(name) LIKE '%' || lower(?) || '%'
             GROUP BY name ORDER BY 2 DESC, name",
        )
        .map_err(fail)?;
    let hits: Vec<String> = stmt
        .query_map([term], |r| {
            Ok(format!(
                "{}({})",
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?
            ))
        })
        .map_err(fail)?
        .collect::<Result<_, _>>()
        .map_err(fail)?;

    let total: i64 = conn
        .query_row("SELECT COUNT(DISTINCT name) FROM skills", [], |r| r.get(0))
        .map_err(fail)?;

    let what = if skill_only { "skill" } else { "mentions" };
    if hits.is_empty() {
        return Ok(format!(
            "{what} \"{term}\" → no skill goes by this name. \
             The pool holds {total} skill spellings \
             (`headhunting distribution skill` for all of them)\n\n"
        ));
    }
    Ok(format!(
        "{what} \"{term}\" → {}  ·  the pool holds {total} skill spellings \
         (`headhunting distribution skill` for all of them)\n\n",
        hits.join(" ")
    ))
}

/// Checks the given cities against the pool and, if one is absent, says which are there.
///
/// # Why this fails rather than answering with nobody
///
/// Searching an absent city returns nobody, which is indistinguishable from "no one in
/// that city fits". The caller then widens the other conditions instead of suspecting a
/// wrong value.
///
/// And this is an easy place to get wrong. `views.sql`'s `location_distribution` reports
/// `positions.location`, whose values drift — `Seoul, KR`, `Greater Seoul Area`, and
/// `Seoul, South Korea` are all the same place. This condition, by contrast, uses the
/// normalized `candidates.city`. **Read the view, pass its value here, and you get
/// nobody.** In a real run the agent hit this and dropped to free-form SQL.
///
/// There are only five cities, so listing them all is one line. That is why it is said
/// where the caller gets stuck rather than attached to every search.
fn check_cities(conn: &rusqlite::Connection, asked: &[String]) -> Result<(), Failure> {
    let fail = |e: rusqlite::Error| Failure::failed(format!("headhunting search: {e}"));

    let mut stmt = conn
        .prepare("SELECT city, COUNT(*) FROM candidates GROUP BY city ORDER BY 2 DESC")
        .map_err(fail)?;
    let have: Vec<(String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(fail)?
        .collect::<Result<_, _>>()
        .map_err(fail)?;

    let missing: Vec<&String> = asked
        .iter()
        .filter(|c| !have.iter().any(|(name, _)| name == *c))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    let listed: Vec<String> = have
        .iter()
        .map(|(name, n)| format!("{name}({n})"))
        .collect();
    Err(Failure::usage(format!(
        "headhunting search: no such city in the pool: {}\n\
         cities in the pool: {}",
        missing
            .iter()
            .map(|c| format!("{c:?}"))
            .collect::<Vec<_>>()
            .join(" "),
        listed.join(" ")
    )))
}

/// The term as FTS5 receives it.
///
/// Quoted because of hyphens. FTS5 reads a bare hyphen as a column qualifier, so
/// `rust-lang` ends in `no such column: lang`. Quoting here is what keeps the instruction
/// from having to warn about it.
fn fts_term(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

fn holes(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",")
}

struct Args {
    /// **Several of these mean all of them must hold**, not the last one.
    ///
    /// It used to be one value that each `--skill` overwrote. A run that needed
    /// `Kubernetes AND MLOps AND PyTorch` wrote the three flags in a row and got back the
    /// rows for `pytorch` alone — a wrong answer indistinguishable from a right one. There
    /// was no way to express the AND at all, so the agent built it by hand from three
    /// searches and checked it with free-form SQL.
    skills: Vec<String>,
    mentions: Vec<String>,
    cities: Vec<String>,
    min_years: Option<f64>,
    name: Option<String>,
    language: Option<String>,
    ids: Vec<String>,
    columns: Vec<String>,
    limit: usize,
}

impl Args {
    fn parse(args: &[&str]) -> Result<Args, Failure> {
        let mut parsed = Args {
            skills: Vec::new(),
            mentions: Vec::new(),
            cities: Vec::new(),
            min_years: None,
            name: None,
            language: None,
            ids: Vec::new(),
            columns: DEFAULT_COLUMNS.iter().map(|s| s.to_string()).collect(),
            limit: DEFAULT_LIMIT,
        };

        let mut it = args.iter().copied();
        let value = |it: &mut dyn Iterator<Item = &str>, flag: &str| {
            it.next()
                .map(str::to_string)
                .ok_or_else(|| Failure::usage(format!("headhunting search: {flag} takes a value")))
        };

        while let Some(arg) = it.next() {
            match arg {
                "--skill" => parsed.skills.push(value(&mut it, "--skill")?),
                "--mentions" => parsed.mentions.push(value(&mut it, "--mentions")?),
                // A person carries one name and one profile language, so a second value
                // cannot also hold. Saying so beats answering with nobody.
                "--name" => {
                    parsed.name = Some(once(parsed.name, value(&mut it, "--name")?, "--name")?)
                }
                "--language" => {
                    parsed.language = Some(once(
                        parsed.language,
                        value(&mut it, "--language")?,
                        "--language",
                    )?)
                }
                // These two already take a comma list, so a second one widens the list
                // rather than replacing it.
                "--city" => {
                    parsed.cities.extend(split(&value(&mut it, "--city")?));
                }
                "--id" => {
                    parsed.ids.extend(split(&value(&mut it, "--id")?));
                }
                "--min-years" => {
                    let raw = value(&mut it, "--min-years")?;
                    parsed.min_years = Some(raw.parse().map_err(|_| {
                        Failure::usage(format!(
                            "headhunting search: --min-years: {raw:?} is not a number"
                        ))
                    })?);
                }
                "--limit" => {
                    let raw = value(&mut it, "--limit")?;
                    parsed.limit = raw.parse().map_err(|_| {
                        Failure::usage(format!(
                            "headhunting search: --limit: {raw:?} is not a number"
                        ))
                    })?;
                }
                "--columns" => {
                    let raw = value(&mut it, "--columns")?;
                    let asked = split(&raw);
                    for name in &asked {
                        if !COLUMNS.iter().any(|(n, _)| n == name) {
                            let all: Vec<&str> = COLUMNS.iter().map(|(n, _)| *n).collect();
                            return Err(Failure::usage(format!(
                                "headhunting search: unknown column {name:?}. Available: {}",
                                all.join(" ")
                            )));
                        }
                    }
                    parsed.columns = asked;
                }
                other => {
                    return Err(Failure::usage(format!(
                        "headhunting search: unknown option {other:?}\n\n{USAGE}"
                    )));
                }
            }
        }

        let has_condition = !parsed.skills.is_empty()
            || !parsed.mentions.is_empty()
            || parsed.name.is_some()
            || parsed.language.is_some()
            || !parsed.cities.is_empty()
            || !parsed.ids.is_empty()
            || parsed.min_years.is_some();
        if !has_condition {
            return Err(Failure::usage(format!(
                "headhunting search: no conditions at all. Left as is, the whole pool comes back\n\n{USAGE}"
            )));
        }
        Ok(parsed)
    }
}

/// Takes a value for a flag that can only hold one.
///
/// The alternative is what this command used to do: assign over the earlier value and say
/// nothing. That is the worst of the three options — the caller gets an answer to a
/// question they did not ask, and nothing on screen distinguishes it from the right one.
fn once(existing: Option<String>, value: String, flag: &str) -> Result<String, Failure> {
    match existing {
        None => Ok(value),
        Some(first) => Err(Failure::usage(format!(
            "headhunting search: {flag} given twice ({first:?} then {value:?}), and only one \
             can hold. Run it once per value and compare the answers"
        ))),
    }
}

/// Split on commas, trim, and drop empty pieces.
fn split(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::{Headhunting, tests::pool};
    use cortex::exec::{ExecCall, ExecResult, Executable};

    async fn run_it(db: &Path, args: &[&str]) -> ExecResult {
        let call = ExecCall {
            name: "headhunting".into(),
            args: std::iter::once("search")
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

    /// The people whose names appear in the answer, header and preamble aside.
    fn named(got: &str) -> Vec<&str> {
        ["Riley", "Casey", "Jordan", "Rowan", "지훈", "Blake"]
            .into_iter()
            .filter(|name| got.contains(name))
            .collect()
    }

    /// `--skill` looks only at the skill list.
    ///
    /// casey wrote "learning Rust" in the headline and has no such skill. If this
    /// condition found them, the headline bait would survive the search stage.
    #[tokio::test]
    async fn skill_looks_only_at_the_skill_list() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust"]).await;
        assert!(
            !got.contains("Casey"),
            "someone with it only in the headline was found: {got}"
        );
        assert!(got.contains("Riley"), "{got}");
        assert!(got.contains("Jordan"), "{got}");
    }

    /// **Repeating a condition used to throw the earlier one away, silently.**
    ///
    /// A real run needed `Kubernetes AND MLOps AND PyTorch` and wrote the three flags in a
    /// row. It got back the rows for the last flag alone — a wrong answer that looked like
    /// a right one. The agent only caught it by comparing counts, then built the
    /// intersection by hand with `comm -12` and checked it against raw SQL.
    #[tokio::test]
    async fn skill_given_twice_means_both() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "postgresql", "--skill", "rust"]).await;
        assert!(got.contains("Riley"), "the one with both is missing: {got}");
        assert!(
            !got.contains("Jordan"),
            "someone with only one of them survived: {got}"
        );
        assert!(!got.contains("Rowan"), "{got}");
    }

    #[tokio::test]
    async fn mentions_given_twice_means_both() {
        let (_dir, db) = pool();
        let got = out(&db, &["--mentions", "postgresql", "--mentions", "rust"]).await;
        assert!(got.contains("Riley"), "{got}");
        assert!(!got.contains("Jordan"), "{got}");
    }

    /// The header has to answer for every term, or a search that caught nothing on one of
    /// them reads as though it caught everything.
    #[tokio::test]
    async fn the_header_reports_each_term() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "postgresql", "--skill", "rust"]).await;
        assert!(got.contains("PostgreSQL"), "{got}");
        assert!(got.contains("Rust"), "{got}");
    }

    /// `--city` is already a comma list, so repeating it widens rather than replaces.
    #[tokio::test]
    async fn city_given_twice_widens() {
        let (_dir, db) = pool();
        let got = out(&db, &["--city", "Seoul", "--city", "Berlin"]).await;
        assert!(got.contains("Berlin") || got.contains("Blake"), "{got}");
        assert!(got.contains("Riley"), "the first city was dropped: {got}");
    }

    /// A person has one name, so two of them cannot both hold. Saying so beats answering
    /// with nobody.
    #[tokio::test]
    async fn name_given_twice_is_a_usage_error() {
        let (_dir, db) = pool();
        let result = run_it(
            &db,
            &["--name", "Riley Calloway", "--name", "Jordan Merrick"],
        )
        .await;
        assert_eq!(result.exit_code, 2);
        let err = String::from_utf8_lossy(&result.stderr);
        assert!(err.contains("--name"), "{err}");
    }

    /// `--mentions` reads the whole profile, which is why casey is found.
    #[tokio::test]
    async fn mentions_looks_at_the_whole_profile() {
        let (_dir, db) = pool();
        let got = out(&db, &["--mentions", "rust"]).await;
        assert!(got.contains("Casey"), "{got}");
    }

    /// `러스트` shares no token with `rust`, so it is not found.
    ///
    /// **This is the axis the pool tests.** If the tool found this person for you,
    /// whether the agent knows to widen a search would never be tested.
    #[tokio::test]
    async fn a_spelling_that_shares_no_token_is_not_found_for_you() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust"]).await;
        assert!(!got.contains("지훈"), "the tool widened it for us: {got}");
    }

    /// What it does report is that spelling drifts, and how many there are in total.
    #[tokio::test]
    async fn the_answer_says_which_spellings_it_caught_and_how_many_there_are() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust"]).await;
        let head = got.lines().next().unwrap();
        assert!(
            head.contains("Rust(3)"),
            "spellings caught, with counts: {head:?}"
        );
        assert!(head.contains("rust-lang(1)"), "{head:?}");
        assert!(
            head.contains("distribution skill"),
            "it has to say how to widen: {head:?}"
        );
    }

    /// **The problem was a view that exists but cannot be used.**
    ///
    /// `views.sql`'s `location_distribution` reports `positions.location` (which drifts:
    /// `Seoul, KR`, `Greater Seoul Area`, `Seoul, South Korea`), while `--city` uses the
    /// normalized `candidates.city`. Read the view, pass `Seoul, KR`, and nobody comes
    /// back — indistinguishable from "no one lives there".
    ///
    /// In a real run the agent hit this wall and dropped to free-form SQL
    /// (`SELECT DISTINCT city FROM candidates`). One failure should be enough to learn
    /// the terrain.
    #[tokio::test]
    async fn a_city_that_is_not_in_the_pool_says_which_ones_are() {
        let (_dir, db) = pool();
        // Exactly what `location_distribution` reports. No comma, so it is checked whole
        // — `Seoul, KR` would split into `Seoul` and `KR` and test something else.
        let result = run_it(&db, &["--city", "Greater Seoul Area"]).await;
        // 2 rather than 1: the value was not understood and the call can be reissued.
        assert_eq!(result.exit_code, 2);
        let err = String::from_utf8_lossy(&result.stderr);
        assert!(
            err.contains("Greater Seoul Area"),
            "it has to name what is absent: {err}"
        );
        assert!(err.contains("Seoul"), "and what is there: {err}");
        assert!(err.contains("Berlin"), "all of it: {err}");
    }

    /// One wrong value among several is still caught. Searching quietly on the rest would
    /// leave it unclear whether a short result came from a typo or from the pool.
    #[tokio::test]
    async fn one_wrong_city_among_several_is_still_caught() {
        let (_dir, db) = pool();
        let result = run_it(&db, &["--city", "Seoul,Gangnam"]).await;
        assert_eq!(result.exit_code, 2);
        assert!(String::from_utf8_lossy(&result.stderr).contains("Gangnam"));
    }

    #[tokio::test]
    async fn city_is_an_exact_match_and_takes_several() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust", "--city", "Seoul,Seongnam"]).await;
        assert!(!got.contains("Blake"), "the Berlin person survived: {got}");
        assert!(got.contains("Riley"), "{got}");
    }

    /// `--min-years` counts the merged span. riley sums to 8.0 years but is really 6.0,
    /// so a 7-year floor keeps them out.
    #[tokio::test]
    async fn min_years_counts_the_merged_tenure_not_the_sum() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust", "--min-years", "7"]).await;
        assert!(
            !got.contains("Riley"),
            "it filtered on the naive sum: {got}"
        );
    }

    /// What decides is shown by default, and not filtered on.
    ///
    /// rowan passes every condition but is not looking; casey cannot be reached. Remove
    /// them in the tool and the agent never learns such people existed, and cannot say in
    /// the shortlist why they were passed over.
    #[tokio::test]
    async fn the_columns_that_decide_are_shown_but_not_filtered() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust"]).await;
        assert!(
            got.contains("Rowan"),
            "the person who isn't looking was dropped: {got}"
        );
        let header = got
            .lines()
            .find(|l| l.starts_with("id"))
            .expect("the header");
        for column in ["years", "open", "contact"] {
            assert!(header.contains(column), "{column} is missing: {header:?}");
        }
    }

    /// **The pair three runs went to free-form SQL for.**
    ///
    /// `read` carries both figures, but comparing five people's tenure through `read`
    /// means receiving five whole profiles. Side by side in the table, an overlap shows
    /// at a glance.
    #[tokio::test]
    async fn naive_years_sits_beside_years_so_an_overlap_is_visible() {
        let (_dir, db) = pool();
        let got = out(
            &db,
            &["--skill", "rust", "--columns", "name,years,naive_years"],
        )
        .await;
        let riley = got.lines().find(|l| l.contains("Riley")).expect("riley");
        assert!(riley.contains("6.0"), "the merged figure: {riley:?}");
        assert!(riley.contains("8.0"), "the summed figure: {riley:?}");
    }

    /// **A count sent the agent back through `read` for every shortlist.**
    ///
    /// Writing a mail needs to know *how* to reach someone, and that lived only in `read`.
    /// All three runs dug it out with free-form SQL instead.
    #[tokio::test]
    async fn contact_shows_the_method_not_a_count() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust"]).await;
        let riley = got.lines().find(|l| l.contains("Riley")).expect("riley");
        assert!(riley.contains("inmail"), "the method: {riley:?}");
    }

    #[tokio::test]
    async fn columns_can_be_chosen() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust", "--columns", "name,city"]).await;
        let header = got
            .lines()
            .find(|l| l.starts_with("id"))
            .expect("the header");
        assert!(header.contains("city"), "{header:?}");
        assert!(
            !header.contains("headline"),
            "a column that was not chosen survived: {header:?}"
        );
    }

    #[tokio::test]
    async fn an_unknown_column_says_what_there_is() {
        let (_dir, db) = pool();
        let result = run_it(&db, &["--skill", "rust", "--columns", "salary"]).await;
        assert_eq!(result.exit_code, 2);
        let err = String::from_utf8_lossy(&result.stderr);
        assert!(err.contains("salary"), "{err:?}");
        assert!(
            err.contains("city"),
            "it has to say what is available: {err:?}"
        );
    }

    /// Narrowing inside a set already found. An earlier run did this by listing 31 ids.
    #[tokio::test]
    async fn id_narrows_the_search_to_a_set_already_found() {
        let (_dir, db) = pool();
        let got = out(
            &db,
            &[
                "--mentions",
                "rust",
                "--id",
                "urn:li:person:riley,urn:li:person:casey",
            ],
        )
        .await;
        assert_eq!(named(&got), vec!["Riley", "Casey"], "{got}");
    }

    /// **A must-have that could not be gated on.**
    ///
    /// `must_haves.json` lists `profile_language` among `backend-seoul-ko`'s must-haves,
    /// and there was no condition for it — the column could be shown but not filtered on.
    #[tokio::test]
    async fn language_gates_on_the_profile_language() {
        let (_dir, db) = pool();
        let got = out(&db, &["--language", "ko"]).await;
        assert!(got.contains("지훈"), "the ko profile: {got}");
        assert!(!got.contains("Riley"), "an en profile survived: {got}");
    }

    #[tokio::test]
    async fn name_finds_everyone_who_carries_it() {
        let (_dir, db) = pool();
        let got = out(&db, &["--name", "지훈 도"]).await;
        assert!(got.contains("지훈"), "{got}");
    }

    #[tokio::test]
    async fn the_id_is_always_there_to_cite() {
        let (_dir, db) = pool();
        let got = out(&db, &["--skill", "rust", "--columns", "name"]).await;
        assert!(got.contains("urn:li:person:riley"), "{got}");
    }

    #[tokio::test]
    async fn a_truncated_answer_says_how_many_there_were() {
        let (_dir, db) = pool();
        let got = out(&db, &["--mentions", "rust", "--limit", "1"]).await;
        assert!(got.contains("of"), "{got}");
    }

    /// With no conditions the whole pool pours out. This stops that happening by accident.
    #[tokio::test]
    async fn no_condition_at_all_is_a_usage_error() {
        let (_dir, db) = pool();
        let result = run_it(&db, &[]).await;
        assert_eq!(result.exit_code, 2);
    }

    #[tokio::test]
    async fn an_unknown_option_is_a_usage_error() {
        let (_dir, db) = pool();
        let result = run_it(&db, &["--seniority", "staff"]).await;
        assert_eq!(result.exit_code, 2);
    }
}
