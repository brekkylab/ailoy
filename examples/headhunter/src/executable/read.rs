//! `read` — the people you picked, in full. Several lines each.

use std::path::Path;

use super::Failure;

pub const USAGE: &str = "\
headhunting read — the people you picked, in full.

usage:
  headhunting read <id…>

Several ids can be given at once, and that is the better way. Reading one at a time
leaves you unable to put people side by side.

For each person:
  profile     name, city, profile language, whether they are open to work, last update
  tenure      the real span with overlap removed. If spans overlap, the naive sum is shown too
  positions   title, company, dates, employment type, workplace type, description in full
  skills      name and endorsement count
  education   degree, field, school, years. Shown only when there is one
  certs       certificate and who issued it. Shown only when there is one
  speaks      the languages they say they speak, with proficiency. **Not the same as the
              profile language** — the mail follows the profile language regardless
  wants       the arrangement they are looking for. `open to work` says only that they are
              looking; this says whether they want what the posting offers
  contacts    how to reach them. If there is none it says so — that means they cannot be reached

Each position carries its company's id after a `#`. Two spellings of one company are not
visible in the name — `Quantile Labs` and `Quantile Labs Inc.` are the same place — so that
number is what tells one person with two profiles from two people.
";

/// One person, whole. Gathered from five tables.
///
/// In an earlier run the agent asked the same people seven times to assemble this —
/// `candidates`, `candidate_tenure`, `positions`, `candidate_brief`, `contacts`,
/// `skills`, and `candidate_brief` again. Folding that into one call is this command's job.
struct Profile {
    id: String,
    name: String,
    headline: String,
    summary: String,
    city: String,
    country: String,
    language: String,
    open_to_work: bool,
    updated: String,
    naive_months: i64,
    real_months: i64,
    positions: Vec<Position>,
    skills: Vec<(String, i64)>,
    /// School, degree, field, and the years. **A degree is a third-party claim**, and one
    /// trap is a person who has one where the practice should be.
    educations: Vec<(String, String, String, i64, i64)>,
    /// Name and issuing authority. Tempting for the same reason as a degree, and more so:
    /// somebody else issued it, so it looks verified.
    certifications: Vec<(String, String)>,
    /// The languages they say they speak, which is **not** `profile_language`. A `ko`
    /// profile listing English as native is a trap: the mail still follows the profile.
    languages: Vec<(String, String)>,
    /// What arrangement they want. `open_to_work` alone says only that they are looking.
    prefs: Option<Prefs>,
    contacts: Vec<(String, String)>,
}

/// What someone is looking for, when they said.
///
/// **`open_to_work` being true does not mean they want this job.** One posting's trap is a
/// candidate whose flag is set while their desired arrangement contradicts the posting —
/// remote against an office three days a week. Reading the flag alone waves them through.
struct Prefs {
    desired_title: String,
    location_type: String,
    desired_location: String,
    start_date: String,
    employment_type: String,
}

struct Position {
    title: String,
    company: String,
    /// The company's own id. **Two spellings of one company are not visible in the
    /// string** — `Quantile Labs` and `Quantile Labs Inc.` are the same place, and only
    /// this says so. A run needed it to be sure two profiles were one person and had to
    /// go through free-form SQL to get it.
    company_id: String,
    description: String,
    start: String,
    end: Option<String>,
    /// Full-time or contract. **When spans overlap, this is what tells concurrent
    /// employment from a gap in the record.**
    employment: String,
    /// On-site or remote. If the posting requires attendance, this has to match.
    workplace: String,
}

pub fn run(db: &Path, args: &[&str]) -> Result<Vec<u8>, Failure> {
    let ids: Vec<&str> = args
        .iter()
        .copied()
        .filter(|a| !a.starts_with('-'))
        .collect();
    if ids.is_empty() {
        return Err(Failure::usage(format!(
            "headhunting read: no id to read\n\n{USAGE}"
        )));
    }

    let conn = super::sql::open(db)?;
    let found =
        fetch(&conn, &ids).map_err(|e| Failure::failed(format!("headhunting read: {e}")))?;

    let mut out = String::new();
    for id in &ids {
        match found.iter().find(|p| p.id == *id) {
            Some(profile) => out.push_str(&render(profile)),
            // Dropping it silently would leave the caller knowing neither that the
            // person is absent nor that the answer got shorter.
            None => out.push_str(&format!("{id}  — not in the pool\n\n")),
        }
    }
    Ok(out.into_bytes())
}

/// One `?` per id, for the `IN` clause.
fn holes(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",")
}

fn fetch(conn: &rusqlite::Connection, ids: &[&str]) -> Result<Vec<Profile>, rusqlite::Error> {
    use rusqlite::params_from_iter;
    let holes = holes(ids.len());

    // Five queries, one each. Asking per person multiplies the round trips by the number
    // of people, and that is how an earlier run reached 17 reads.
    let mut profiles: Vec<Profile> = conn
        .prepare(&format!(
            "SELECT b.id, b.name, b.headline, c.summary, b.city, b.country,
                    b.profile_language, b.open_to_work, b.last_updated_at,
                    COALESCE(t.naive_months, 0), COALESCE(t.real_months, 0)
             FROM candidate_brief b
             JOIN candidates c ON c.id = b.id
             LEFT JOIN candidate_tenure t ON t.id = b.id
             WHERE b.id IN ({holes})"
        ))?
        .query_map(params_from_iter(ids), |r| {
            Ok(Profile {
                id: r.get(0)?,
                name: r.get(1)?,
                headline: r.get(2)?,
                summary: r.get(3)?,
                city: r.get(4)?,
                country: r.get(5)?,
                language: r.get(6)?,
                open_to_work: r.get::<_, i64>(7)? != 0,
                updated: r.get(8)?,
                naive_months: r.get(9)?,
                real_months: r.get(10)?,
                positions: Vec::new(),
                skills: Vec::new(),
                educations: Vec::new(),
                certifications: Vec::new(),
                languages: Vec::new(),
                prefs: None,
                contacts: Vec::new(),
            })
        })?
        .collect::<Result<_, _>>()?;

    let mut push = |id: String, f: &mut dyn FnMut(&mut Profile)| {
        if let Some(p) = profiles.iter_mut().find(|p| p.id == id) {
            f(p);
        }
    };

    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, title, company_name, description,
                start_year, start_month, end_year, end_month,
                employment_type, workplace_type, company_urn
         FROM positions WHERE candidate_id IN ({holes})
         ORDER BY candidate_id, start_year DESC, start_month DESC"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let position = Position {
            title: r.get(1)?,
            company: r.get(2)?,
            description: r.get(3)?,
            start: ym(r.get(4)?, r.get(5)?),
            end: match (r.get::<_, Option<i64>>(6)?, r.get::<_, Option<i64>>(7)?) {
                (Some(y), m) => Some(ym(y, m.unwrap_or(1))),
                // `end_year IS NULL` means current; the convention is stated in `views.sql`.
                (None, _) => None,
            },
            employment: r.get(8)?,
            workplace: r.get(9)?,
            company_id: short_org(&r.get::<_, String>(10)?),
        };
        push(id, &mut |p| p.positions.push(position_of(&position)));
    }

    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, name, endorsement_count FROM skills
         WHERE candidate_id IN ({holes}) ORDER BY candidate_id, endorsement_count DESC"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let pair = (r.get::<_, String>(1)?, r.get::<_, i64>(2)?);
        push(id, &mut |p| p.skills.push(pair.clone()));
    }

    // The four tables this command used to leave out. Each carries material a judgment
    // needs, and each was reached through free-form SQL in a real run because it was not
    // here.
    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, school_name, degree_name, field_of_study, start_year, end_year
         FROM educations WHERE candidate_id IN ({holes}) ORDER BY candidate_id, end_year DESC"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let row = (
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, i64>(4)?,
            r.get::<_, i64>(5)?,
        );
        push(id, &mut |p| p.educations.push(row.clone()));
    }

    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, name, authority FROM certifications
         WHERE candidate_id IN ({holes}) ORDER BY candidate_id, name"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let row = (r.get::<_, String>(1)?, r.get::<_, String>(2)?);
        push(id, &mut |p| p.certifications.push(row.clone()));
    }

    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, name, proficiency FROM languages
         WHERE candidate_id IN ({holes}) ORDER BY candidate_id, name"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let row = (r.get::<_, String>(1)?, r.get::<_, String>(2)?);
        push(id, &mut |p| p.languages.push(row.clone()));
    }

    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, desired_title, location_type, desired_location,
                start_date, employment_type
         FROM open_to_work_prefs WHERE candidate_id IN ({holes}) ORDER BY candidate_id"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let row = Prefs {
            desired_title: r.get(1)?,
            location_type: r.get(2)?,
            desired_location: r.get(3)?,
            start_date: r.get(4)?,
            employment_type: r.get(5)?,
        };
        push(id, &mut |p| {
            p.prefs = Some(Prefs {
                desired_title: row.desired_title.clone(),
                location_type: row.location_type.clone(),
                desired_location: row.desired_location.clone(),
                start_date: row.start_date.clone(),
                employment_type: row.employment_type.clone(),
            })
        });
    }

    let mut stmt = conn.prepare(&format!(
        "SELECT candidate_id, method, note FROM contacts
         WHERE candidate_id IN ({holes}) ORDER BY candidate_id"
    ))?;
    let mut rows = stmt.query(params_from_iter(ids))?;
    while let Some(r) = rows.next()? {
        let id: String = r.get(0)?;
        let pair = (r.get::<_, String>(1)?, r.get::<_, String>(2)?);
        push(id, &mut |p| p.contacts.push(pair.clone()));
    }

    Ok(profiles)
}

/// `Position` is not `Clone`, so a value to move into the closure is built here.
fn position_of(p: &Position) -> Position {
    Position {
        title: p.title.clone(),
        company: p.company.clone(),
        description: p.description.clone(),
        start: p.start.clone(),
        end: p.end.clone(),
        employment: p.employment.clone(),
        workplace: p.workplace.clone(),
        company_id: p.company_id.clone(),
    }
}

/// The tail of `urn:li:organization:1694752`.
///
/// The whole urn on every position line would cost more width than it earns; what has to
/// be comparable is the number.
fn short_org(urn: &str) -> String {
    urn.rsplit(':').next().unwrap_or(urn).to_string()
}

fn ym(year: i64, month: i64) -> String {
    format!("{year:04}-{month:02}")
}

fn render(p: &Profile) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "{}  {}  ·  {}, {}  ·  {}  ·  {}  ·  updated {}\n",
        p.id,
        p.name,
        p.city,
        p.country,
        p.language,
        if p.open_to_work {
            "open to work"
        } else {
            "not open to work"
        },
        p.updated
    ));
    out.push_str(&format!("  headline   {}\n", p.headline));
    out.push_str(&format!("  summary    {}\n", p.summary));

    // Writing the same number twice for someone with no overlap would make the person
    // who does have one blend in.
    let overlap = p.naive_months - p.real_months;
    let years = |m: i64| format!("{:.1} years", m as f64 / 12.0);
    if overlap > 0 {
        out.push_str(&format!(
            "  tenure     {} months ({}) once merged. Summed it is {}, but {} overlap\n",
            p.real_months,
            years(p.real_months),
            p.naive_months,
            overlap
        ));
    } else {
        out.push_str(&format!(
            "  tenure     {} months ({})\n",
            p.real_months,
            years(p.real_months)
        ));
    }

    if !p.positions.is_empty() {
        out.push_str("  positions\n");
        for job in &p.positions {
            out.push_str(&format!(
                "    {} ~ {}  {} · {} #{}  ({} · {})\n",
                job.start,
                job.end.as_deref().unwrap_or("present"),
                job.title,
                job.company,
                job.company_id,
                job.employment,
                job.workplace
            ));
            out.push_str(&format!("      {}\n", job.description));
        }
    }

    if !p.skills.is_empty() {
        let listed: Vec<String> = p
            .skills
            .iter()
            .map(|(name, n)| format!("{name}({n})"))
            .collect();
        out.push_str(&format!("  skills     {}\n", listed.join(", ")));
    }

    // These four are empty for most people, so an empty heading on every profile would
    // bury the ones who do have a row.
    if !p.educations.is_empty() {
        let listed: Vec<String> = p
            .educations
            .iter()
            .map(|(school, degree, field, from, to)| {
                format!("{degree}, {field} · {school} ({from}~{to})")
            })
            .collect();
        out.push_str(&format!("  education  {}\n", listed.join(" | ")));
    }
    if !p.certifications.is_empty() {
        let listed: Vec<String> = p
            .certifications
            .iter()
            .map(|(name, authority)| format!("{name} — {authority}"))
            .collect();
        out.push_str(&format!("  certs      {}\n", listed.join(" | ")));
    }
    // Deliberately not merged with the profile language on the first line. They answer
    // different questions, and the mail follows the profile language whatever this says.
    if !p.languages.is_empty() {
        let listed: Vec<String> = p
            .languages
            .iter()
            .map(|(name, level)| format!("{name} ({level})"))
            .collect();
        out.push_str(&format!("  speaks     {}\n", listed.join(", ")));
    }
    if let Some(w) = &p.prefs {
        // Someone who wants remote work has "Remote" in both fields, and "Remote in
        // Remote" reads as a mistake.
        let where_ = if w.location_type == w.desired_location {
            w.location_type.clone()
        } else {
            format!("{} in {}", w.location_type, w.desired_location)
        };
        out.push_str(&format!(
            "  wants      {} · {} · from {} · {}\n",
            w.desired_title, where_, w.start_date, w.employment_type
        ));
    }

    // No row means there is no way to reach them. Left blank it would be indistinguishable
    // from a query that was never run.
    if p.contacts.is_empty() {
        out.push_str("  contacts   none — there is no way to reach this person\n");
    } else {
        for (method, note) in &p.contacts {
            out.push_str(&format!("  contacts   {method} · {note}\n"));
        }
    }

    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::{Headhunting, tests::pool};
    use cortex::exec::{ExecCall, ExecResult, Executable};

    async fn run_it(db: &Path, args: &[&str]) -> ExecResult {
        let call = ExecCall {
            name: "headhunting".into(),
            args: std::iter::once("read")
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

    /// **`open_to_work` being true is not the whole answer.**
    ///
    /// `open_to_work_prefs` says what arrangement they actually want, and one posting's
    /// trap is exactly a person whose flag is true while their desired arrangement
    /// contradicts the posting. `read` did not show this table, so a run had to reach for
    /// free-form SQL three times to get it.
    #[tokio::test]
    async fn what_they_want_is_shown_beside_the_open_flag() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley"]).await;
        assert!(
            got.contains("Remote"),
            "the desired arrangement is missing: {got}"
        );
    }

    /// A degree and a certificate are third-party claims, and two traps turn on accepting
    /// one in place of evidence. They cannot be weighed while they are invisible.
    #[tokio::test]
    async fn education_and_certification_are_shown() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:jordan", "urn:li:person:rowan"]).await;
        assert!(got.contains("Computer Science"), "no education: {got}");
        assert!(
            got.contains("Certified Kubernetes Administrator"),
            "no certification: {got}"
        );
    }

    /// The languages someone speaks and the language their profile is written in are
    /// different facts. The instruction says the mail follows `profile_language` whatever
    /// the fluency entry says — which can only be resisted once it is on screen.
    #[tokio::test]
    async fn spoken_languages_sit_beside_the_profile_language() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:jihun"]).await;
        assert!(got.contains("NATIVE_OR_BILINGUAL"), "{got}");
        assert!(
            got.contains("  ko  "),
            "the profile language is still there: {got}"
        );
    }

    /// **Two spellings of one company are not visible in the string.**
    ///
    /// A run had to pull `company_urn` through free-form SQL to be sure two profiles were
    /// the same person. The fixture's `Pinehurst` and `Pinehurst Systems` are one company.
    #[tokio::test]
    async fn the_company_id_is_shown_so_two_spellings_can_be_told_apart() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:casey", "urn:li:person:jordan"]).await;
        assert!(got.contains("Pinehurst"), "{got}");
        assert_eq!(
            got.matches("13").count(),
            2,
            "the same company id has to appear under both spellings: {got}"
        );
    }

    /// Most people have none of these. An empty heading on every profile would bury the
    /// people who do have one.
    #[tokio::test]
    async fn a_section_with_no_rows_is_left_out() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:blake"]).await;
        assert!(!got.contains("education"), "{got}");
        assert!(!got.contains("certifications"), "{got}");
    }

    #[tokio::test]
    async fn a_profile_opens_with_the_id_so_it_can_be_cited() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley"]).await;
        assert!(got.starts_with("urn:li:person:riley"), "{got:?}");
        assert!(got.contains("Riley Calloway"), "{got:?}");
        assert!(got.contains("Seoul"), "{got:?}");
    }

    /// **The test closest to why this command exists.**
    ///
    /// riley's two spans overlap by 24 months: summed they are 96, merged they are 72.
    /// Show only the merged figure and the naive-sum mistake can never happen, which
    /// removes the reason that person is in the pool at all; show only the sum and the
    /// tool is lying.
    #[tokio::test]
    async fn overlapping_tenure_shows_both_numbers_and_the_overlap() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley"]).await;
        assert!(got.contains("96"), "the summed figure is missing: {got:?}");
        assert!(got.contains("72"), "the merged figure is missing: {got:?}");
        assert!(got.contains("24"), "the overlap is missing: {got:?}");
    }

    /// Writing the same number twice for someone with no overlap would make the person
    /// who does have one blend in.
    #[tokio::test]
    async fn tenure_that_does_not_overlap_is_one_number() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:rowan"]).await;
        let tenure = got
            .lines()
            .find(|l| l.contains("tenure"))
            .expect("the tenure line");
        assert!(
            !tenure.contains("overlap"),
            "it claims an overlap where there is none: {tenure:?}"
        );
    }

    /// Position descriptions have to be carried. They are the only place that separates
    /// what a headline claims from what the person actually did.
    #[tokio::test]
    async fn positions_carry_their_descriptions_in_full() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:jordan"]).await;
        assert!(
            got.contains("Ran the payments pipeline on Java and Kafka."),
            "{got:?}"
        );
        // jordan's skill list has Rust with no evidence in any description. That has to
        // be visible.
        assert!(
            got.contains("Rust"),
            "the skills come through as skills: {got:?}"
        );
    }

    /// **These missing fields are what called the back door.**
    ///
    /// In a real run `query` was used twice and both were `positions` lookups, after
    /// `employment_type` and `workplace_type`. The first tells concurrent employment
    /// apart; the second says whether the posting's attendance requirement is met. Both
    /// are material for the judgment, and `read` was not carrying them.
    #[tokio::test]
    async fn positions_carry_how_the_work_was_done() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley"]).await;
        assert!(
            got.contains("CONTRACT"),
            "the employment type is missing: {got}"
        );
        assert!(
            got.contains("HYBRID"),
            "the workplace type is missing: {got}"
        );
    }

    #[tokio::test]
    async fn a_current_role_is_marked_rather_than_left_blank() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:rowan"]).await;
        assert!(
            got.contains("present"),
            "a current role has to show as current: {got:?}"
        );
    }

    #[tokio::test]
    async fn skills_carry_their_endorsement_counts() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley"]).await;
        assert!(got.contains("Rust(31)"), "{got:?}");
    }

    /// Contactability is expressed as **the absence of a row**. Left blank it would be
    /// indistinguishable from a lookup that was never made.
    #[tokio::test]
    async fn no_contact_row_is_said_out_loud() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:casey"]).await;
        let line = got
            .lines()
            .find(|l| l.contains("contacts"))
            .expect("the contacts line");
        assert!(line.contains("none"), "{line:?}");
    }

    #[tokio::test]
    async fn several_people_come_back_in_one_answer() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley", "urn:li:person:casey"]).await;
        assert!(got.contains("Riley Calloway"), "{got:?}");
        assert!(got.contains("Casey Ashby"), "{got:?}");
    }

    /// The order asked for is kept: the agent calls in rank order and reads in that order.
    #[tokio::test]
    async fn the_order_asked_for_is_the_order_answered() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:casey", "urn:li:person:riley"]).await;
        let casey = got.find("Casey Ashby").expect("casey");
        let riley = got.find("Riley Calloway").expect("riley");
        assert!(casey < riley, "{got:?}");
    }

    /// Dropping an absent id silently would leave the caller knowing neither that the
    /// person is missing nor that the answer got shorter.
    #[tokio::test]
    async fn an_id_that_is_not_there_is_named() {
        let (_dir, db) = pool();
        let got = out(&db, &["urn:li:person:riley", "urn:li:person:nobody"]).await;
        assert!(got.contains("nobody"), "{got:?}");
        assert!(
            got.contains("Riley Calloway"),
            "the rest is still answered: {got:?}"
        );
    }

    #[tokio::test]
    async fn no_id_at_all_is_a_usage_error() {
        let (_dir, db) = pool();
        let result = run_it(&db, &[]).await;
        assert_eq!(result.exit_code, 2);
    }
}
