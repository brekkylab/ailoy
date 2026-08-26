"""Whether the machine-readable must-have conditions still match the data.

**Numbers written by hand rot when the data changes.** That failure repeated in this
project: `58/600` became `56/600` while three comments still said the old figure, and with
no definition nobody could re-derive it. That is why the conditions are machine-readable,
and this script is what re-derives them.

    python3 eval/check_must_haves.py
"""

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
DB = HERE.parent / "data" / "headhunter.db"


def where(cond: dict) -> tuple[str, list]:
    """One condition as a SQL fragment. `b` is the `candidate_brief` alias."""
    kind, value = cond["kind"], cond["value"]
    if kind == "city_in":
        return f"b.city IN ({','.join('?' * len(value))})", list(value)
    if kind == "profile_language":
        return "b.profile_language = ?", [value]
    if kind == "real_months_at_least":
        # `candidate_brief` carries only real_years; months come from the tenure view.
        return "(SELECT real_months FROM candidate_tenure t WHERE t.id = b.id) >= ?", [value]
    if kind == "skill_matches":
        return ("EXISTS (SELECT 1 FROM skills s WHERE s.candidate_id = b.id "
                "AND LOWER(s.name) LIKE ?)", [value])
    if kind == "skill_any_of":
        # **Every spelling is accepted.** The dataset writes one skill as `Rust`,
        # `rust-lang`, `러스트`, and `Tokio` (spec §3.4), and the instruction tells the
        # agent to widen accordingly. A gate narrowed to `LIKE '%rust%'` would drop the
        # very people that instructed behaviour finds.
        #
        # It is also the division of labour between gating and reading. The gate lets
        # people **through broadly**, and "did they actually do it" is settled by reading
        # the position descriptions — in a real run the agent rejected one Tokio holder
        # exactly that way, on the description rather than the skill tag.
        return (f"EXISTS (SELECT 1 FROM skills s WHERE s.candidate_id = b.id "
                f"AND s.name IN ({','.join('?' * len(value))}))", list(value))
    if kind == "skills_all":
        return (f"(SELECT COUNT(DISTINCT s.name) FROM skills s WHERE s.candidate_id = b.id "
                f"AND s.name IN ({','.join('?' * len(value))})) = {len(value)}", list(value))
    raise SystemExit(f"unknown condition kind: {kind!r}")


def main() -> int:
    con = sqlite3.connect(DB)
    spec = json.loads((HERE / "jd" / "must_haves.json").read_text())
    violations = []

    for jd, entry in spec.items():
        if jd.startswith("_"):
            continue
        clauses, params = [], []
        for cond in entry["conditions"]:
            sql, args = where(cond)
            clauses.append(sql)
            params += args
        n = con.execute(
            f"SELECT COUNT(*) FROM candidate_brief b WHERE {' AND '.join(clauses)}", params
        ).fetchone()[0]

        said = entry["expected_qualified"]
        k = entry["k"]
        mark = " " if n == said else "X"
        shape = "emit fewer" if n < k else "must narrow"
        print(f"  {mark} {jd:22} measured {n:3}  recorded {said:3}  k={k:2}  {shape}")
        if n != said:
            violations.append(f"{jd}: the conditions yield {n} but expected_qualified is {said}")
        # This inequality is what decides what the posting tests. Flip it and
        # `run_eval.py`'s criteria change wholesale.
        if (n < k) != (said < k):
            violations.append(f"{jd}: measured and recorded fall on opposite sides of k={k} — a different thing is being tested")

        # Also check the posting file exists. Without it there is nothing to give the agent.
        if not (HERE / "jd" / f"{jd}.md").exists():
            violations.append(f"{jd}: no eval/jd/{jd}.md")

    if violations:
        print(f"\n{len(violations)} violations")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("\nno violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
