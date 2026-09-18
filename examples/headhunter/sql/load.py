"""Loads the JSON into SQLite.

# Why Python

This is a preparation step that runs once before the example does. Neither the agent nor
ailoy calls it. It reads `data/candidates.json` — the pool as committed — and writes the
database the `headhunting` command opens.

# journal_mode is decided here

`cortex-execs/sqlite` opens someone else's database read-only and forbids WAL. But the
journal mode is a property of **the file header**, not of the connection, so the reader
cannot guarantee it — that falls to whoever creates the file (Plan A, "what Plan C has to
own", item 2). It is set to `DELETE` explicitly.
"""

import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

# The month a run is taken to happen in. A position with no end date runs until here, so
# this is what "still there" resolves to when tenure is totalled. The pool was generated
# against this value, and `views.sql` carries it again as a SQL literal —
# `assert_as_of_matches` below is what keeps those two from drifting apart in silence.
AS_OF = (2026, 8)


def assert_as_of_matches() -> None:
    """Whether the literal in `views.sql` matches `AS_OF` above.

    **This is a place that goes wrong quietly, with no error.** `views.sql` fills the end
    of a current role with `2026*12+8`. If that diverges from `AS_OF`, the answer key's
    tenure figures and the view's arithmetic disagree while neither SQL nor Python
    complains — and overlapping tenure is one of the things this pool is built to test.

    Rendering the SQL from a template would remove the second copy, but it would also make
    `views.sql` a generated artifact rather than a file people read and edit. At this size
    the check buys what rendering would, and the file stays readable by hand.
    """
    expected = f"{AS_OF[0]}*12+{AS_OF[1]}"
    raw = (HERE / "views.sql").read_text()
    # **Comments are stripped first.** The same literal also appears in the explanatory
    # comments, so searching the whole text would let the check be satisfied by a comment
    # when the SQL was changed and the comment was not (or the reverse) — it fails open.
    #
    # This was missed even when the check was "verified by breaking it": the mutation was
    # `sed 's/2026\*12+8/2027*12+8/'`, which changed **the comment and the SQL together**.
    # A real mistake happens on one side only.
    code = re.sub(r"--.*", "", raw)
    if expected not in code:
        raise SystemExit(
            f"the SQL in views.sql does not contain {expected!r} — it diverges from the "
            f"AS_OF={AS_OF} in this file. The answer key's tenure figures and the view's "
            f"arithmetic will disagree silently"
        )
    # Also check it is in the place that fills the end of a current role. The same
    # arithmetic appearing incidentally elsewhere must not satisfy this.
    if not re.search(rf"COALESCE\([^)]*\)\s*,\s*{re.escape(expected)}\s*\)", code):
        raise SystemExit(
            f"views.sql contains {expected!r} but not in the `COALESCE(..., {expected})` "
            f"that fills the end of a current role"
        )


def load(db_path: Path) -> None:
    candidates = json.loads((DATA / "candidates.json").read_text())
    narration_path = DATA / "narration.json"
    narration = json.loads(narration_path.read_text()) if narration_path.exists() else {}

    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    # Must not be WAL. See the module docstring.
    con.execute("PRAGMA journal_mode = DELETE")
    con.executescript((HERE / "schema.sql").read_text())
    con.executescript((HERE / "views.sql").read_text())

    for c in candidates:
        prose = narration.get(c["id"], {})
        con.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                c["id"], c["first_name"], c["last_name"], c["headline"],
                prose.get("summary", c.get("summary", "")),
                c["city"], c["country"], c["industry"], c["job_function"],
                c["seniority"], c["profile_language"], int(c["open_to_work"]),
                c["connections_count"], c["last_updated_at"], c["public_profile_url"],
            ),
        )
        descriptions = prose.get("descriptions", [])
        for i, p in enumerate(c["positions"]):
            con.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    c["id"], i, p["title"], p["company_name"], p["company_urn"],
                    p["company_size"], p["employment_type"], p["workplace_type"],
                    p["location"],
                    descriptions[i] if i < len(descriptions) else p.get("description", ""),
                    p["start_year"], p["start_month"],
                    p.get("end_year"), p.get("end_month"),
                ),
            )
        for s in c["skills"]:
            con.execute(
                "INSERT INTO skills VALUES (?,?,?)",
                (c["id"], s["name"], s["endorsement_count"]),
            )
        for e in c.get("educations", []):
            con.execute(
                "INSERT INTO educations VALUES (?,?,?,?,?,?)",
                (c["id"], e["school_name"], e["degree_name"], e["field_of_study"],
                 e["start_year"], e["end_year"]),
            )
        for t in c.get("certifications", []):
            con.execute(
                "INSERT INTO certifications VALUES (?,?,?)",
                (c["id"], t["name"], t["authority"]),
            )
        for lang in c.get("languages", []):
            con.execute(
                "INSERT INTO languages VALUES (?,?,?)",
                (c["id"], lang["name"], lang["proficiency"]),
            )
        for pref in c.get("open_to_work_prefs", []):
            con.execute(
                "INSERT INTO open_to_work_prefs VALUES (?,?,?,?,?,?)",
                (c["id"], pref["desired_title"], pref["location_type"],
                 pref["desired_location"], pref["start_date"], pref["employment_type"]),
            )
        for contact in c.get("contacts", []):
            con.execute(
                "INSERT INTO contacts VALUES (?,?,?)",
                (c["id"], contact["method"], contact["note"]),
            )

    # FTS5 is filled from the normalized tables. A candidate's several rows concatenate
    # into one document, because `MATCH 'rust'` has to find that person whether it sits in
    # a skill or a headline.
    con.execute("""
        INSERT INTO candidate_fts (id, headline, summary, titles, descriptions, skill_names)
        SELECT c.id, c.headline, c.summary,
               (SELECT group_concat(title, ' ') FROM positions WHERE candidate_id = c.id),
               (SELECT group_concat(description, ' ') FROM positions WHERE candidate_id = c.id),
               (SELECT group_concat(name, ' ') FROM skills WHERE candidate_id = c.id)
        FROM candidates c
    """)
    con.commit()
    counts = {
        t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("candidates", "positions", "skills", "contacts", "candidate_fts")
    }
    con.close()
    print(f"wrote {db_path}")
    for t, n in counts.items():
        print(f"  {t:16} {n}")


def main() -> None:
    assert_as_of_matches()
    load(DATA / "headhunter.db")


if __name__ == "__main__":
    main()
