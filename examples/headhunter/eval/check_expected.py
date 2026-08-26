"""Whether every id in the answer key exists, and whether it is what it claims to be.

**One typo makes the scoring silently meaningless.** Put an id that does not exist in
`must_not_appear` and that check passes forever — whatever the agent does, that id cannot
appear in a shortlist.

    python3 eval/check_expected.py
"""

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
DB = HERE.parent / "data" / "headhunter.db"
TRUTH = HERE.parent / "data" / "ground_truth.json"


def main() -> int:
    con = sqlite3.connect(DB)
    ids = {r[0] for r in con.execute("SELECT id FROM candidates")}
    truth = {t["id"]: t for t in json.loads(TRUTH.read_text())}
    must = json.loads((HERE / "jd" / "must_haves.json").read_text())
    violations = []

    for path in sorted((HERE / "expected").glob("*.json")):
        exp = json.loads(path.read_text())
        jd = exp["jd"]

        # Gather every cited id in one place
        named: dict[str, str] = {}
        for i in exp.get("acceptable_top_k", []):
            named[i] = "acceptable_top_k"
        for section in ("must_not_appear", "traps_that_must_be_caught",
                        "controls_that_must_not_be_rejected"):
            for entry in exp.get(section, []):
                if isinstance(entry, dict) and entry.get("id"):
                    named[entry["id"]] = section

        missing = {i: where for i, where in named.items() if i not in ids}
        for i, where in missing.items():
            violations.append(f"{jd}: {i} in {where} is not in the database")

        # Whether the labels match the answer key — catches a valid id pointing at
        # someone else
        for entry in exp.get("must_not_appear", []) + exp.get("traps_that_must_be_caught", []):
            i, said = entry.get("id"), entry.get("trap")
            if not i or i not in truth or said is None:
                continue
            actual = truth[i]["trap"]
            if actual != said:
                violations.append(
                    f"{jd}: {i} is called {said!r} but the answer key says {actual!r}")
        for entry in exp.get("controls_that_must_not_be_rejected", []):
            i, said = entry.get("id"), entry.get("control_for")
            if not i or i not in truth:
                continue
            actual = truth[i]["control_for"]
            if actual != said:
                violations.append(
                    f"{jd}: {i} is called a control for {said!r} but the answer key says {actual!r}")

        # k has to match must_haves.json. Diverge and the two files test different things
        if jd in must and exp["k"] != must[jd]["k"]:
            violations.append(
                f"{jd}: k={exp['k']} in expected differs from k={must[jd]['k']} in must_haves")
        # `expected_fewer_than_k` has to match the measured figure too
        if jd in must:
            actual_fewer = must[jd]["expected_qualified"] < must[jd]["k"]
            if exp.get("expected_fewer_than_k") != actual_fewer:
                violations.append(
                    f"{jd}: expected_fewer_than_k={exp.get('expected_fewer_than_k')} but "
                    f"measured it is {must[jd]['expected_qualified']} < {must[jd]['k']} = {actual_fewer}")

        print(f"  {'  ' if not missing else 'X '} {jd:22} {len(named):2} cited  "
              f"absent ids: {sorted(missing) or 'none'}")

    if violations:
        print(f"\n{len(violations)} violations")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("\nno violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
