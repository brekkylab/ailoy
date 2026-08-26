"""What the FTS5 index actually recalls.

Plan B's `variants.fts5_recall` does not call `sqlite3`. It matches hand-split tokens
against **one field**. This script puts the same question to the real index: every
candidate, every field, through the query the agent will actually write.

    python3 eval/check_index.py
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent
DB = HERE / "data" / "headhunter.db"

# The tokens Plan B's `variants.needle_for` settled on. They are restated here to avoid
# importing `datagen` — an example that depends on the generator cannot ship alone.
# If the values diverge, this script surfaces it as a recall difference.
NEEDLES = {
    "Rust": "rust",
    "Kubernetes": "kubernetes",
    "Distributed Systems": "distributed",
    "Senior Backend Engineer": "backend",
    "Backend Engineer": "backend",
    "Seoul, KR": "seoul",
    "Tokyo, JP": "tokyo",
}

# Where each axis reads "its value" from. The approximation looks at this one field only.
#
# **The `location` axis is not here.** The five fields `candidate_fts` indexes do not
# include a position's `location`. That is right: location has an exact column in
# `candidates.city`, and `WHERE city IN (…)` is more precise than full-text search. FTS is
# for free text.
#
# What was wrong is Plan B's measurement. `variants.fts5_recall` measured `Seoul, KR` and
# `Tokyo, JP` and called the result "FTS5 recall", but that axis is not searched through
# FTS at all. What it actually measured was "the share of location strings containing a
# seoul token", which is not search recall. Measured: `MATCH 'seoul'` brings back **one**
# of 402 people, and that one has Seoul in their summary by chance.
FTS_AXES = {
    "Rust": "SELECT candidate_id, name FROM skills",
    "Kubernetes": "SELECT candidate_id, name FROM skills",
    "Distributed Systems": "SELECT candidate_id, name FROM skills",
    "Senior Backend Engineer": "SELECT candidate_id, title FROM positions",
    "Backend Engineer": "SELECT candidate_id, title FROM positions",
}

# Axes found by column. Checked with `WHERE`, not the index.
COLUMN_AXES = {
    "Seoul, KR": ("city", ("Seoul", "Seongnam")),
    "Tokyo, JP": ("city", ("Tokyo",)),
}


def tokens(text: str) -> set[str]:
    """As `unicode61` splits it. The same rule as Plan B's `variants._tokens`."""
    return {t for t in re.split(r"[^\w]+", text.lower(), flags=re.UNICODE) if t}


def match(con: sqlite3.Connection, needle: str) -> set[str]:
    """The ids this token brings back from the real index.

    The table name is spelled on both sides of `MATCH` — an alias dies with
    `no such column`.
    """
    return {
        r[0] for r in con.execute(
            "SELECT id FROM candidate_fts WHERE candidate_fts MATCH ?", (needle,))
    }


def main() -> int:
    con = sqlite3.connect(DB)
    truth = {t["id"]: t for t in json.loads((HERE / "data" / "ground_truth.json").read_text())}
    violations: list[str] = []

    # ── (1) recall per axis: the approximation against the real index ────
    print("FTS axes — the approximation (one field) against the real index (five)\n")
    for canonical, sql in FTS_AXES.items():
        needle = NEEDLES[canonical]
        approx = {cid for cid, value in con.execute(sql) if value and needle in tokens(value)}
        real = match(con, needle)
        gained, lost = real - approx, approx - real
        print(f"  {canonical:26} needle={needle:12} approx {len(approx):3}  real {len(real):3}"
              f"  gained +{len(gained):3}  lost -{len(lost)}")
        if lost:
            # If the approximation finds someone the real index does not, the
            # tokenizer assumption is wrong.
            violations.append(
                f"{canonical}: {len(lost)} people the approximation finds and the real "
                f"index does not — the tokenizer assumption is wrong: {sorted(lost)[:3]}")
        if not gained:
            violations.append(
                f"{canonical}: the five fields gained nobody — check that `summary` and "
                f"`descriptions` actually went into `candidate_fts`")

    # ── (1b) column axes: found by column, not by index ──────────────────
    #
    # That these axes are not reached by FTS is design, not a defect. What is checked is
    # **whether the column side actually brings people back**; at zero, a posting's
    # location condition would find nobody.
    print("\ncolumn axes — found with WHERE, not the index\n")
    for canonical, (column, values) in COLUMN_AXES.items():
        placeholders = ",".join("?" * len(values))
        n = con.execute(
            f"SELECT COUNT(*) FROM candidates WHERE {column} IN ({placeholders})", values
        ).fetchone()[0]
        by_fts = len(match(con, NEEDLES[canonical]))
        print(f"  {canonical:26} {column} IN {values} → {n:3}"
              f"   (for reference: MATCH '{NEEDLES[canonical]}' gives {by_fts} — this axis is not indexed)")
        if n == 0:
            violations.append(
                f"{canonical}: even by {column} it is zero — a posting's location condition "
                f"would find nobody")

    # ── (2) per posting: are the answers and traps reachable by search ───
    must = json.loads((HERE / "eval" / "jd" / "must_haves.json").read_text())
    print("\nper posting — are the answers and search-stage traps in the index\n")
    for path in sorted((HERE / "eval" / "expected").glob("*.json")):
        exp = json.loads(path.read_text())
        jd = exp["jd"]
        needles = []
        for cond in must.get(jd, {}).get("conditions", []):
            kind = cond["kind"]
            if kind == "skills_all":
                needles += [NEEDLES.get(s, s.split()[0].lower()) for s in cond["value"]]
            elif kind == "skill_matches":
                needles.append(cond["value"].strip("%").lower())
            elif kind == "skill_any_of":
                # The list of spelling variants. The **first token** of each is the needle
                # that finds it — `rust` for `rust-lang`, `러스트` for `러스트`, `tokio`
                # for `Tokio`.
                for name in cond["value"]:
                    toks = [t for t in re.split(r"[^\w]+", name.lower()) if t]
                    if toks:
                        needles.append(toks[0])
            elif kind == "city_in":
                needles += [c.split(",")[0].lower() for c in cond["value"]]
            elif kind in ("real_months_at_least", "profile_language"):
                pass  # a condition filtered by column, not by the index
            else:
                violations.append(f"{jd}: check_index does not know the condition kind {kind!r}")
        if not needles:
            violations.append(f"{jd}: no needle could be derived from must_haves.json")
            continue

        # The search an agent actually makes: sweep the must-haves broadly with OR, then
        # narrow. Narrowing with AND would make it impossible to tell what the index
        # **can** reach from what it **does**.
        reachable: set[str] = set()
        for needle in needles:
            reachable |= match(con, needle)

        must_reach = {e["id"] for e in exp.get("controls_that_must_not_be_rejected", [])}
        must_reach |= set(exp.get("acceptable_top_k", []))
        unreachable = must_reach - reachable

        traps = {e["trap"] for e in exp.get("traps_that_must_be_caught", []) if e.get("trap")}
        trap_ids = {i for i, t in truth.items() if t["trap"] in traps and t["jd"] == jd}
        trap_unreachable = trap_ids - reachable

        print(f"  {jd:22} needles={','.join(sorted(set(needles)))}")
        print(f"    {'reachable by index':26} {len(reachable)}")
        print(f"    {'answers + controls':26} {len(must_reach):3}  unreachable {sorted(unreachable) or 'none'}")
        print(f"    {'search-stage traps':26} {len(trap_ids):3}  unreachable {sorted(trap_unreachable) or 'none'}")

        if unreachable:
            violations.append(
                f"{jd}: {len(unreachable)} answers/controls are not in the index — "
                f"scoring is meaningless: {sorted(unreachable)}")
        if trap_unreachable:
            violations.append(
                f"{jd}: {len(trap_unreachable)} search-stage traps are not in the index — "
                f"those traps are dead and scoring reads it as 'the agent avoided them': "
                f"{sorted(trap_unreachable)}")

    if violations:
        print(f"\n{len(violations)} violations")
        for v in violations:
            print(f"  - {v}")
        print("\n**When a violation appears, fix the posting, not the data.** A trap missing")
        print("from the index means the spelling variants were spread too aggressively, and the")
        print("right fix is widening that posting's needle to the spelling that person carries.")
        print("Changing the data would mean re-running all of Plan B's measurements.")
        return 1
    print("\nno violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
