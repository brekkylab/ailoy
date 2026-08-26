"""Tests for the dataset itself.

The unit tests check that the generator's functions behave. This file reads the **finished
data** and catches the cases where the generator behaved and the data still came out wrong.

The two that matter most happen silently. A trap can die — shift a date until the overlap
is gone and the data still looks plausible and the tests still pass, but there is nothing
left to catch. And the spelling drift can go too far, leaving a search that finds nothing,
which is not an error either.
"""

import itertools
import json
import re
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from common.dates import AS_OF, merge_spans, months, naive_sum
from common.names import DENYLIST
from headhunter import variants
from headhunter.fixtures import TRAPS, core
from headhunter.profile import TRAP_ASSIGNMENTS
from headhunter.schema import gap_months
from headhunter.truth import spans_for

DATA = Path(__file__).parent.parent.parent / "data"

# The core is counted from `fixtures.core()`; the background is the remainder.
POPULATION = 600

# **This is not the same check as `tests/test_variants.py`.** That one asks whether the
# tables *can* produce a recall in band — it measures the design. This one asks whether the
# generated data's recall *is* in band. Leave the tables alone, change how the generator
# applies them or edit the data by hand, and the tests pass while the data drifts.
RECALL_BAND = (0.85, 0.95)

# How many search-stage traps should be in one posting's results.
SEARCH_STAGE_BAND = (2, 3)

# Columns checked for a fingerprint by exact value. Each holds 2–8 values, so
# `SELECT DISTINCT` enumerates them all. The pair scan combines from this same list.
_EXACT_COLUMNS = (
    "industry",
    "job_function",
    "seniority",
    "profile_language",
    "city",
    "country",
    "open_to_work",
)

# Known exceptions to the pair scan, written as `(label, value)` — silencing a whole axis
# would silence its future fingerprints too.
#
# The Korean-profile-in-Berlin pair belongs to one person, the `location-mismatch` trap.
# All three ways of removing it break something else: adding `ko` to the generator weights
# shifts 45 people's names, adding a background person makes the population 601 with no
# prose written for them, and switching the trap to `en` contradicts the Korean prose
# already written for them.
#
# The fingerprint also buys little. That trap is defined by location alone, so the language
# pairing is incidental, and it is a judge-stage trap — the agent sees this person through
# the search regardless.
_PAIR_EXEMPT = frozenset({
    ("(profile_language, city)", ("ko", "Berlin")),
    ("(profile_language, country)", ("ko", "DE")),
})

# The must-have threshold, in months. `check_traps_alive` and `measure_search_results` both
# use it; move one and the assertion about what a filter catches parts from the filter.
MUST_HAVE_MONTHS = 48

# "Small enough to read through." **Widened from 40–60 to 40–70, deliberately.**
#
# The band stands in for a continuous property — whether the set can be read through — and
# 63 is not distinguishable from 60 for that purpose. So the ceiling moved rather than the
# data. Had the measurement been 120 the data would have to change instead; that number
# does alter the property.
READING_BAND = (40, 70)

# **Derived from `TrapSpec.stage`, never written out.** `stage` has no default, so adding a
# trap forces the decision; a second list here would let someone change `stage` and forget
# this one.
SEARCH_STAGE_TRAPS = frozenset(
    name for name, spec in TRAPS.items() if spec.stage == "search"
)
JUDGE_STAGE_TRAPS = frozenset(
    name for name, spec in TRAPS.items() if spec.stage == "judge"
)


def check_integrity(candidates: list[dict], truth: list[dict]) -> list[str]:
    """Referential integrity, and that nothing here names anything real."""
    violations = []

    ids = {c["id"] for c in candidates}
    if len(ids) != len(candidates):
        violations.append(f"duplicate ids: {len(candidates) - len(ids)}")

    truth_ids = {t["id"] for t in truth}
    if ids != truth_ids:
        violations.append(f"candidates and ground_truth hold different ids: {len(ids ^ truth_ids)}")

    for c in candidates:
        name = f"{c['first_name']} {c['last_name']}"
        if name in DENYLIST:
            violations.append(f"denylisted person: {name}")
        for e in c.get("educations", []):
            # Committed data must not read as fictional alumni of a real university.
            if e["school_name"].split()[0] in DENYLIST:
                violations.append(f"denylisted school: {e['school_name']}")
        for p in c["positions"]:
            if p["company_name"].split()[0] in DENYLIST:
                violations.append(f"denylisted company: {p['company_name']}")

    for c in candidates:
        if not c["positions"]:
            violations.append(f"{c['id']}: no positions")

    # One size per company. Drawing the size per position once left 142 of 143 urns with
    # more than one, and one company existing at seven sizes at once. **That broke the
    # `inflated-title` trap**: it is "a CXO at a 1-10 company" against "a CXO at a large
    # one", and the comparison is gone if the trap's company is 201-500 elsewhere.
    #
    # Neither column's distribution shows it. Both looked fine — the relation between them
    # was broken.
    size_by_urn: dict[str, set[str]] = {}
    holders_by_urn: dict[str, set[str]] = {}
    current_by_urn: dict[str, set[str]] = {}
    stem_by_urn: dict[str, set[str]] = {}
    urn_by_stem: dict[str, set[str]] = {}
    for c in candidates:
        for p in c["positions"]:
            urn = p["company_urn"]
            size_by_urn.setdefault(urn, set()).add(p["company_size"])
            holders_by_urn.setdefault(urn, set()).add(c["id"])
            if p.get("end_year") is None:
                current_by_urn.setdefault(urn, set()).add(c["id"])
            base = _company_base(p["company_name"])
            stem_by_urn.setdefault(urn, set()).add(base)
            urn_by_stem.setdefault(base, set()).add(urn)
    for urn, sizes in size_by_urn.items():
        if len(sizes) > 1:
            violations.append(
                f"{urn}: size splits across {sorted(sizes)} — one company, one size"
            )

    # **urn ↔ company name, both ways.** The spelling drifts on purpose and the urn is the
    # only evidence two spellings are one company. Break that and `duplicate-profile` loses
    # what connects its two profiles — differing company names are its definition.
    #
    # Measured: changing one position's urn produced zero violations before this was here.
    for urn, bases in stem_by_urn.items():
        if len(bases) > 1:
            violations.append(
                f"{urn}: names split across {sorted(bases)} — one urn is one company"
            )
    for base, urns in urn_by_stem.items():
        if len(urns) > 1:
            violations.append(
                f"the company {base!r} has urns {sorted(urns)} — with two urns for one "
                f"company the urn is no longer evidence of identity"
            )

    # No more people employed than the band allows. The check above can pass and this still
    # break — eleven people at a consistently `1-10` company.
    caps = {
        "1-10": 10,
        "11-50": 50,
        "51-200": 200,
        "201-500": 500,
        "501-1,000": 1000,
        "1,001-5,000": 5000,
        "5,001-10,000": 10000,
        "10,001+": 10**9,
    }
    for urn, sizes in size_by_urn.items():
        if len(sizes) != 1:
            continue  # already reported above
        (size,) = sizes
        # **Currently employed, not everyone who ever passed through.** Over fifteen years
        # of careers, 25 people cycling through a ten-person company is normal; counting
        # cumulatively raises a false alarm on correct data.
        headcount = len(current_by_urn.get(urn, ()))
        if size in caps and headcount > caps[size]:
            violations.append(
                f"{urn}: size {size} with {headcount} currently employed — over the band"
            )

    return violations


# The company name with its legal suffix stripped. The drift only adds and removes those.
#
# **Not the first word.** Names are a head plus a tail, so `Quantile Labs` and
# `Quantile Systems` share a first word and are different companies — keying on it produced
# eight violations on correct data.
_LEGAL_SUFFIX = re.compile(r"\s+(?:Inc\.?|Corporation|Corp\.?|Co\.,\s*Ltd\.?|Ltd\.?|LLC|GmbH|AG|KK)$")


def _company_base(name: str) -> str:
    """The name with any legal suffix stripped."""
    prev = None
    while prev != name:
        prev, name = name, _LEGAL_SUFFIX.sub("", name).strip()
    return name


def _core_ids(truth: list[dict]) -> set[str]:
    """The hand-written 65, defined once.

    This expression lived in two places, and with two the fingerprint check and the recall
    check could call different sets the core — both passing.

    Taken from `truth` rather than `fixtures.core()` because this file reads the committed
    artifacts. `check_core_survives` compares the two definitions.
    """
    return {t["id"] for t in truth if t["trap"] or t["control_for"] or t["verdict"]}


def _profile(candidate: dict) -> SimpleNamespace:
    """Just enough shape for `spans_for`.

    Not `Position(**p)`: a current role has no end keys at all, and one unknown key in
    hand-edited data would raise instead of reporting a violation.
    """
    return SimpleNamespace(
        positions=[
            SimpleNamespace(
                start_year=p["start_year"],
                start_month=p["start_month"],
                end_year=p.get("end_year"),
                end_month=p.get("end_month"),
            )
            for p in candidate["positions"]
        ]
    )


def check_the_truth_recomputes(candidates: list[dict], truth: list[dict]) -> list[str]:
    """Recomputes the answer key's numbers from `candidates.json` and compares.

    **The trap-survival check was measuring the answer key against itself.**
    `check_traps_alive` reads `naive_months > real_months`, and both come from
    `ground_truth.json`. Measured: flattening the `overlapping-tenure` trap's dates to
    1990–1991 still reported the trap alive, because the key was untouched. The two files
    had never been compared.

    Recomputing the spans here defends **all 14 trap kinds** at once — whatever each
    trap's own condition checks, a profile and a key that disagree are caught.

    **`AS_OF` is verified here.** `spans_for` fills a current role's end with it, so a key
    computed against a different `AS_OF` puts everyone with a current role out. The other
    half — comparing against `views.sql` — belongs to whoever loads that file.
    """
    violations = []
    by_truth = {t["id"]: t for t in truth}
    now = months(*AS_OF)

    for c in candidates:
        cid = c["id"]
        t = by_truth.get(cid)
        if t is None:
            continue  # `check_integrity` already reports the id mismatch

        # Date validity. There was no check here: a position starting in 2030 and an
        # update stamped 2030-01 both produced zero violations.
        well_formed = True
        for p in c["positions"]:
            start = months(p["start_year"], p["start_month"])
            if start > now:
                violations.append(
                    f"{cid}: position starts {p['start_year']}-{p['start_month']:02d}, "
                    f"after AS_OF ({AS_OF[0]}-{AS_OF[1]:02d})"
                )
            if ("end_year" in p) != ("end_month" in p):
                # A broken pair puts `None` into `spans_for`'s arithmetic, which is why
                # the recomputation below is skipped for this person.
                violations.append(
                    f"{cid}: one of end_year/end_month is present — a current role has neither"
                )
                well_formed = False
                continue
            if "end_year" not in p:
                continue
            end = months(p["end_year"], p["end_month"])
            if end < start:
                violations.append(f"{cid}: position ends before it starts")
            if end > now:
                violations.append(
                    f"{cid}: position ends {p['end_year']}-{p['end_month']:02d}, after AS_OF"
                )

        year, month = (int(x) for x in c["last_updated_at"].split("-")[:2])
        if months(year, month) > now:
            violations.append(
                f"{cid}: last_updated_at is {c['last_updated_at']}, after AS_OF"
            )

        # Recompute the spans. The key and the profile have to agree.
        if well_formed:
            spans = spans_for(_profile(c))
            for label, computed, recorded in (
                ("naive_months", naive_sum(spans), t["naive_months"]),
                ("real_months", merge_spans(spans), t["real_months"]),
                ("gap_months", gap_months(spans), t["gap_months"]),
            ):
                if computed != recorded:
                    violations.append(
                        f"{cid}: the key says {label} is {recorded}, the profile "
                        f"recomputes to {computed} — the two files have parted"
                    )

        # Values the key copies straight from the profile. Same kind of drift, no cost.
        for label, computed, recorded in (
            ("job_function", c["job_function"], t["job_function"]),
            ("seniority", c["seniority"], t["seniority"]),
            ("profile_language", c["profile_language"], t["profile_language"]),
            ("open_to_work", c["open_to_work"], t["open_to_work"]),
            ("has_contact", bool(c.get("contacts")), t["has_contact"]),
            ("skills_listed", sorted(s["name"] for s in c["skills"]), t["skills_listed"]),
        ):
            if computed != recorded:
                violations.append(
                    f"{cid}: the key's {label} differs from the profile "
                    f"({recorded!r} vs {computed!r})"
                )

    return violations


def check_core_survives(candidates: list[dict], truth: list[dict]) -> list[str]:
    """Whether the hand-written 65 are still in the committed data.

    `check_integrity` only asks whether the two files hold the **same** ids. Remove one
    person from both and the sets still match — measured, deleting a core person produced
    zero violations.

    The fixture tests assert the 65 and their breakdown, but they measure
    `fixtures.core()`. The tests guard the code; this file guards the committed artifacts.

    **No expected number is written here.** The 65 and the 17/11/37 come from
    `fixtures.core()`; pinning them would make growing the core by one a silent failure.
    """
    violations = []
    if len(candidates) != POPULATION:
        violations.append(
            f"{len(candidates)} candidates — the population is {POPULATION}"
        )

    # Labels come from two places: the hand-written `core()`, and `TRAP_ASSIGNMENTS`, which
    # adds nobody and so appears in no fixture. Expecting only `core()` would report every
    # assigned trap as missing.
    FIELDS = ("trap", "control_for", "verdict", "pair_with", "jd")
    expected_labels = {
        c.id: {f: getattr(c, f"_{f}", None) for f in FIELDS} for c in core()
    }
    # Half the assigned people come from the 37 judged. A control has to be an ordinary
    # person rather than another trap, and background people who are open to work and
    # reachable are rare enough that four could not be found (measured: zero in Tokyo).
    # So `verdict` and `pair_with` keep their fixture values and only the label is added.
    for cid, assigned in TRAP_ASSIGNMENTS.items():
        base = dict(expected_labels.get(cid, {f: None for f in FIELDS}))
        if base["jd"] is not None and base["jd"] != assigned.get("jd"):
            violations.append(
                f"{cid}: assigned to {assigned.get('jd')!r} but the fixture says "
                f"{base['jd']!r} — a person moved between postings is misplaced"
            )
        base["trap"] = assigned.get("trap")
        base["control_for"] = assigned.get("control_for")
        base["jd"] = assigned.get("jd")
        expected_labels[cid] = base

    data_core = _core_ids(truth)
    for missing in sorted(expected_labels.keys() - data_core):
        violations.append(f"{missing}: core, but unlabelled in the data — the fixture is gone")
    for extra in sorted(data_core - expected_labels.keys()):
        violations.append(f"{extra}: the data calls this core, and no fixture does")

    # Labels too. By id alone, one half of the `duplicate-profile` pair can lose its trap
    # label and still pass — they are core either way, so the id set does not move.
    by_truth = {t["id"]: t for t in truth}
    for cid in sorted(expected_labels.keys() & data_core):
        for field in FIELDS:
            expected = expected_labels[cid][field]
            if by_truth[cid][field] != expected:
                violations.append(
                    f"{cid}: the key's {field} is {by_truth[cid][field]!r}, "
                    f"expected {expected!r}"
                )

    # The 17/11/37 breakdown, counted from the expectations rather than written down.
    for label, field in (("traps", "trap"), ("controls", "control_for"), ("judged", "verdict")):
        expected = sum(1 for v in expected_labels.values() if v[field])
        actual = sum(1 for t in truth if t[field])
        if expected != actual:
            violations.append(
                f"{actual} {label} in the data — expected {expected}"
            )

    return violations


def check_traps_alive(candidates: list[dict], truth: list[dict]) -> list[str]:
    """Whether each trap kind is still alive.

    **`TrapSpec.checks` is not read here.** It is prose and cannot be evaluated; the per-trap
    checks below and the fixture tests are where those conditions live in code.
    """
    violations = []
    by_id = {c["id"]: c for c in candidates}
    by_trap: dict[str, list[dict]] = {}
    for t in truth:
        if t["trap"]:
            by_trap.setdefault(t["trap"], []).append(t)

    # A typo in `stage` puts the trap in neither set, so `measure_search_results` quietly
    # stops counting it — and that count is the pass criterion.
    for name in sorted(set(TRAPS) - (SEARCH_STAGE_TRAPS | JUDGE_STAGE_TRAPS)):
        violations.append(
            f"trap {name!r} has stage {TRAPS[name].stage!r} — "
            f'must be "search" or "judge"'
        )
    # Moving a trap between stages changes this, and that changes what `SEARCH_STAGE_BAND`
    # rests on — seven search-stage people across three postings.
    if len(SEARCH_STAGE_TRAPS) != 5:
        violations.append(
            f"{len(SEARCH_STAGE_TRAPS)} search-stage trap kinds — there should be 5"
        )

    for name in TRAPS:
        if name not in by_trap:
            violations.append(f"trap {name!r} is gone")

    for t in by_trap.get("overlapping-tenure", []):
        if t["naive_months"] <= t["real_months"]:
            violations.append(f"{t['id']}: no overlap — the trap is dead")

    pair = by_trap.get("rank-inversion-pair", [])
    if len(pair) != 2:
        violations.append(f"the inversion pair has {len(pair)} people")
    else:
        a, b = sorted(pair, key=lambda t: t["naive_months"], reverse=True)
        if not (a["naive_months"] > b["naive_months"] and a["real_months"] < b["real_months"]):
            violations.append("the order does not invert")
        if min(a["real_months"], b["real_months"]) < MUST_HAVE_MONTHS:
            violations.append("one of the inversion pair misses the must-have — a filter removes them")

    # A control must not have become a trap.
    for t in truth:
        if t["control_for"] == "overlapping-tenure" and t["naive_months"] != t["real_months"]:
            violations.append(f"{t['id']}: a control whose spans overlap")

    return violations


def check_no_column_fingerprints_the_core(
    candidates: list[dict], truth: list[dict]
) -> list[str]:
    """No single column value may select the core.

    The core is hand-written and the background generated, which makes it easy for a
    constant to end up on all 65 — `industry="Computer Software"` for everyone. Then the
    agent can find them by that column instead of by reading, and gets the right answer for
    the wrong reason. Finding fingerprints one at a time is not enough: a review flagged
    `job_function` and three more of the same kind were sitting there.

    **The condition is containment, not equality.** An earlier version asked whether the
    holders *equalled* the core and missed a real one: every background position was
    `FULL_TIME` and only two core traps held `CONTRACT`, so
    `WHERE employment_type != 'FULL_TIME'` named those two. Selecting *part* of the core is
    the same leak — selecting one person is worth more than selecting 65, because what the
    agent wants is who to read next.

    Multi-row columns like positions and skills are included; there is no reason to assume
    a fingerprint lives at the candidate level. The `id` prefix is checked too — an ident
    like `rank-inv-a` reaches both `id` and `public_profile_url`.

    **What this misses** is any shape not named as a projection. Every background headline
    once had exactly one comma, so `WHERE headline NOT LIKE '%,%'` selected 14 core people
    while this check passed. Free text cannot be enumerated this way.

    **The signal that a projection is missing is the background being uniform on some
    component** — all positions `FULL_TIME`, all dates ending `-01`. A generator that pins
    an axis has made that axis a fingerprint.
    """
    violations = []
    core_ids = _core_ids(truth)
    if not core_ids:
        return ["cannot identify the core — ground_truth has no trap/control_for/verdict"]

    def flag(
        label: str,
        pairs: list[tuple[object, str]],
        exempt: frozenset[tuple[str, object]] = frozenset(),
    ) -> None:
        """`pairs` is (value, candidate id). A value held only by core people is a
        fingerprint. `exempt` names `(label, value)` — exempting a whole label would
        silence that axis's future fingerprints too.
        """
        by_value: dict[object, set[str]] = {}
        for value, cid in pairs:
            by_value.setdefault(value, set()).add(cid)
        for value, ids in by_value.items():
            if ids and ids <= core_ids and (label, value) not in exempt:
                violations.append(
                    f"all {len(ids)} people with {label}={value!r} are core — a "
                    f"fingerprint. Someone in the background needs this value too"
                )

    # **Exact values only for low-cardinality columns.** With 2–8 values, `SELECT DISTINCT`
    # enumerates them, so a value held only by the core is one the agent can find.
    for column in _EXACT_COLUMNS:
        flag(column, [(c.get(column), c["id"]) for c in candidates])

    # **Pairs of columns too.** Each distribution can look right while the relation between
    # Skipped above cardinality 12, on the same reasoning as the single columns: what does
    # not enumerate cannot be discovered. Twenty-one pairs from seven columns costs nothing,
    # and `SELECT DISTINCT a, b` is a query an agent writes without being told to.
    for a, b in itertools.combinations(_EXACT_COLUMNS, 2):
        pairs = [((c.get(a), c.get(b)), c["id"]) for c in candidates]
        if len({v for v, _ in pairs}) > 12:
            continue
        flag(f"({a}, {b})", pairs, exempt=_PAIR_EXEMPT)

    # **High-cardinality columns are not checked by exact value — they match by chance.**
    #
    # Including `connections_count` produced 50 violations that were noise: of 484 values
    # held by one person each, 48 belonged to the core where 52 were expected. Fewer than
    # chance. Using `connections_count = 641` would require already knowing 641 matters.
    #
    # Low-cardinality **projections** instead. The fingerprint is in the pattern, not the
    # value: `last_updated_at` has 61 values but its day component effectively has two, and
    # that is where the real leak was — all 535 background dates ended `-01`.
    for column, projections in (
        (
            "last_updated_at",
            (
                ("day", lambda v: v.split("-")[2]),
                ("year", lambda v: v.split("-")[0]),
                ("month", lambda v: v.split("-")[1]),
            ),
        ),
        (
            "connections_count",
            (
                ("hundreds", lambda v: v // 100),
                ("ends in zero", lambda v: v % 10 == 0),
            ),
        ),
    ):
        for label, project in projections:
            flag(
                f"{column} {label}",
                [(project(c[column]), c["id"]) for c in candidates],
            )

    # Several rows per candidate. The one fingerprint actually found — `CONTRACT` — was at
    # the position level, not the candidate level.
    for column in ("employment_type", "workplace_type", "company_size"):
        flag(column, [(p[column], c["id"]) for c in candidates for p in c["positions"]])
    flag("skill", [(s["name"], c["id"]) for c in candidates for s in c["skills"]])

    # **A count is a projection too**, and a list length is not a column value. A real one:
    # the background held 3–6 skills and nine core people held 2, so
    # `GROUP BY candidate_id HAVING COUNT(*) = 2` named those nine.
    for label, count_of in (
        ("skill count", lambda c: len(c["skills"])),
        ("position count", lambda c: len(c["positions"])),
        ("contact count", lambda c: len(c.get("contacts", []))),
    ):
        flag(label, [(count_of(c), c["id"]) for c in candidates])
    flag(
        "contact",
        [(t["method"], c["id"]) for c in candidates for t in c.get("contacts", [])],
    )

    # A trap name must not reach the id.
    leaking = ("trap", "rank-inv", "dup", "control", "inflated", "stale", "bait")
    for c in candidates:
        ident = c["id"].rsplit(":", 1)[-1].lower()
        for word in leaking:
            if word in ident:
                violations.append(f"{c['id']}: the ident contains {word!r} — the answer leaks")

    return violations


def check_controls_are_not_traps(candidates: list[dict], truth: list[dict]) -> list[str]:
    """Whether any control satisfies a trap condition.

    **This is not "does it differ on one axis".** The fixture tests ask whether a control
    differs from *its* trap on exactly one axis — whether the axis holds. This asks whether
    a control has accidentally become one of the *other* traps. A `stale-profile` control
    that is recently updated and happens to carry a 14-month gap is an `employment-gap`
    trap, not a clean comparison.

    Here rather than in the tests because this reads the **committed data**. The tests guard
    the code; hand-edited data is caught only here.

    **Ten conditions are machine-decidable right now.** `headline-bait` and
    `skills-without-evidence` live in `description`, which the prose layer fills.

    `korean-only-profile` is also out, deliberately. The trap is not "the profile is
    Korean" but "a Korean-only profile among an English-language posting's results".
    Korean is normal inside `backend-seoul-ko`, so testing `profile_language == 'ko'` alone
    catches that posting's other controls by accident — measured, two of them. **The
    condition is relative to the posting**, and which posting is outside this function's
    input. Adding `t["jd"] != "backend-seoul-ko"` would be right today and silently wrong
    the moment a second Korean posting exists.
    """
    violations = []
    by_id = {c["id"]: c for c in candidates}

    # Trap conditions decidable from the data alone. `(name, condition)`; true means the
    # person is that trap, and every one has to be false for a control.
    def is_trap(name: str, c: dict, t: dict) -> bool:
        if name == "overlapping-tenure":
            return t["naive_months"] > t["real_months"]
        if name == "employment-gap":
            return t["gap_months"] >= 12
        if name == "stale-profile":
            return c["last_updated_at"] < "2025-01-01"
        if name == "no-contact":
            return not c.get("contacts")
        if name == "strong-but-not-open":
            return not c["open_to_work"]
        if name == "korean-only-profile":
            # **Relative to the posting, so undecidable alone.** Korean is normal inside
            # `backend-seoul-ko`, and testing the language alone catches that posting's
            # other controls by accident. `run_eval.py` knows the posting; this does not.
            return False
        if name == "inflated-title":
            return c["seniority"] == "CXO" and c["positions"][0]["company_size"] == "1-10"
        return False  # conditions living in the prose, in a pairing, or in the posting

    for t in truth:
        target = t.get("control_for")
        if not target:
            continue
        c = by_id[t["id"]]
        for name in TRAPS:
            # Its own trap's condition is absent by construction — that is the axis. What
            # is checked here is whether it became one of the **others**.
            if name == target:
                continue
            if is_trap(name, c, t):
                violations.append(
                    f"{t['id']}: a control for {target!r} that also satisfies {name!r} — "
                    f"not a comparison, another trap"
                )
    return violations


def check_profile_tables_are_populated(
    candidates: list[dict], truth: list[dict]
) -> list[str]:
    """Whether the side tables actually hold values, and hold consistent ones.

    Without this the four tables passed while empty. Structural validation asks whether the
    fields match the schema, and zero rows is a valid answer to that.
    """
    violations: list[str] = []
    bands = {
        "educations": (0.70, 0.90),
        "certifications": (0.20, 0.40),
        "languages": (0.60, 0.80),
    }
    for field, (lo, hi) in bands.items():
        ratio = sum(1 for c in candidates if c.get(field)) / len(candidates)
        if not lo <= ratio <= hi:
            violations.append(f"{field} is held by {ratio:.2f}, outside {lo}–{hi}")

    open_people = [c for c in candidates if c["open_to_work"]]
    ratio = sum(1 for c in open_people if c.get("open_to_work_prefs")) / len(open_people)
    if not 0.60 <= ratio <= 0.80:
        violations.append(f"preferences held by {ratio:.2f}, outside 0.60–0.80")

    name_of = {"en": "English", "ko": "Korean", "ja": "Japanese"}
    for c in candidates:
        if not c["open_to_work"] and c.get("open_to_work_prefs"):
            violations.append(f"{c['id']}: not open to work but carries preferences")
        if c.get("educations") and c.get("positions"):
            last_end = max(e["end_year"] for e in c["educations"])
            first_start = min(p["start_year"] for p in c["positions"])
            if last_end > first_start:
                violations.append(
                    f"{c['id']}: graduated {last_end}, after starting work in {first_start}"
                )
        langs = {l["name"] for l in c.get("languages", [])}
        want = name_of[c["profile_language"]]
        if langs and want not in langs:
            violations.append(f"{c['id']}: profile language {want} is not in the language list")

    return violations


def check_recall_of_the_generated_data(candidates: list[dict], truth: list[dict]) -> list[str]:
    """How much of the drift a needle finds in the generated data.

    **Measured over the background only.** Measuring all 600 gave 0.744 for
    `Backend Engineer`; split, the background was 0.824 and the core 0.613. The core's
    `backend-seoul-ko` people typed their titles in Korean by hand, and Korean shares no
    token with an English needle, so every one is a miss.

    That is not a defect — it is the intended difficulty. What this check guards against is
    **too much drift**, and the drift is what the generator injects into the background.

    **Judged by interval, not by point estimate.** The background samples run 51–188. At
    `n=51` and `p≈0.85` the standard error is about 0.05, so the 95% interval is wider than
    the 0.10 band itself: 0.824 and 0.85 are not distinguishable with that sample. A
    violation is raised only when the interval misses the band entirely — calling an
    indistinguishable difference a violation makes a check that flips between runs.
    """
    violations = []
    low, high = RECALL_BAND
    core_ids = _core_ids(truth)
    background = [c for c in candidates if c["id"] not in core_ids]

    for canonical, accessor, field in variants.canonicals():
        wanted = set(accessor(canonical))
        used = [v for c in background for v in field(c) if v in wanted]
        n = len(used)
        if n < 30:
            print(f"  (skipped) {canonical}: background sample of {n} — under 30")
            continue
        recall = variants.fts5_recall(canonical, used)
        # A 95% interval. Overlapping the band means no judgment.
        half = 1.96 * math.sqrt(max(recall * (1 - recall), 0.01) / n)
        lo, hi = recall - half, recall + half
        overlaps = hi >= low and lo <= high
        mark = "  " if overlaps else "!!"
        print(
            f"  {mark} {canonical:26} n {n:4}  recall {recall:.3f} "
            f"±{half:.3f}  [{lo:.3f}, {hi:.3f}]"
        )
        if not overlaps:
            violations.append(
                f"{canonical!r}: the background recall's 95% interval is "
                f"[{lo:.3f}, {hi:.3f}], which does not meet {low}-{high} "
                f"(needle={variants.needle_for(canonical)!r}, n={n})"
            )
    return violations


def measure_search_results(candidates: list[dict], truth: list[dict]) -> dict[str, dict]:
    """What each posting's search surfaces.

    The posting is not parsed; FTS5 is imitated with the skill tokens it asks for. The real
    search belongs to the example app — counting is all that is needed here.

    - `agent_reads` — the size of the set the agent's own query returns. **This is what the
      reading-size band judges**, without the `extra` predicate below.
    - `matched` — the per-posting set narrowed by `extra`. What `misplaced` and
      `search_stage` are counted against.
    - `search_stage` — how many of those are **search-stage** traps. A pass criterion.
    - `misplaced` — traps placed on another posting that appear here. Anything but zero
      means a trap and its control have parted, and the comparison no longer holds.
    - `density` — the share of results that are traps. **Recorded, not judged.** A
      judge-stage trap is someone the search found correctly.
    """
    by_id = {t["id"]: t for t in truth}
    out: dict[str, dict] = {}
    # backend-rust and backend-seoul-ko ask for the same skills, so tokens alone cannot
    # separate them — one needle makes both matched sets identical and every trap placed on
    # one shows as misplaced on the other. Language is what actually separates them, so the
    # predicate is (token, extra condition).
    #
    # **The reading-size band is not applied to that predicate's output.** No query in this
    # example filters on language; the agent runs `MATCH <needle> AND real_years >= 4`.
    # Judging the language-split share would let the raw match grow to 120 while the band
    # stays green. Language separates the postings; `agent_reads` measures the size.
    for jd, needle, extra in (
        ("backend-rust", "rust", lambda c: c["profile_language"] == "en"),
        ("ml-platform-tokyo", "pytorch", None),
        ("backend-seoul-ko", "rust", lambda c: c["profile_language"] == "ko"),
        ("blockchain-solidity", "solidity", None),
    ):
        hits = [c for c in candidates if _matches(c, needle)]
        # What the agent runs: `MATCH <needle> AND real_years >= 4`. No language predicate.
        agent_reads = sum(
            1 for c in hits if by_id[c["id"]]["real_months"] >= MUST_HAVE_MONTHS
        )
        matched = [c for c in hits if extra is None or extra(c)]
        if not matched:
            out[jd] = {
                "agent_reads": agent_reads,
                "matched": 0,
                "traps": 0,
                "search_stage": 0,
                "density": 0.0,
                "misplaced": 0,
            }
            continue
        traps = [c for c in matched if by_id[c["id"]]["trap"]]
        search_stage = [c for c in traps if by_id[c["id"]]["trap"] in SEARCH_STAGE_TRAPS]
        misplaced = sum(1 for c in traps if by_id[c["id"]]["jd"] not in (jd, None))
        out[jd] = {
            "agent_reads": agent_reads,
            "matched": len(matched),
            "traps": len(traps),
            "search_stage": len(search_stage),
            "density": len(traps) / len(matched),
            "misplaced": misplaced,
        }
    return out


def _matches(candidate: dict, needle: str) -> bool:
    """Whether FTS5 would find this candidate by `needle`."""
    haystack = [candidate["headline"], candidate.get("summary", "")]
    haystack += [s["name"] for s in candidate["skills"]]
    haystack += [p["title"] for p in candidate["positions"]]
    haystack += [p["description"] for p in candidate["positions"]]
    tokens = set()
    for text in haystack:
        current = []
        for ch in text:
            if ch.isalnum():
                current.append(ch.lower())
            elif current:
                tokens.add("".join(current))
                current = []
        if current:
            tokens.add("".join(current))
    return needle in tokens


def main() -> None:
    candidates = json.loads((DATA / "candidates.json").read_text())
    truth = json.loads((DATA / "ground_truth.json").read_text())

    violations = (
        check_integrity(candidates, truth)
        + check_the_truth_recomputes(candidates, truth)
        + check_core_survives(candidates, truth)
        + check_traps_alive(candidates, truth)
        + check_no_column_fingerprints_the_core(candidates, truth)
        + check_controls_are_not_traps(candidates, truth)
        + check_recall_of_the_generated_data(candidates, truth)
        + check_profile_tables_are_populated(candidates, truth)
    )
    search = measure_search_results(candidates, truth)

    print(f"{len(candidates)} candidates, {len(truth)} answer-key entries")
    read_lo, read_hi = READING_BAND
    print(
        f"\nsearch results (pass: 2-3 search-stage traps, 0 misplaced, "
        f"backend-rust reads {read_lo}-{read_hi}):"
    )
    print(
        f"  {'posting':24} {'reads':>6} {'match':>6} {'search':>8} "
        f"{'traps':>8} {'density':>8}  misplaced"
    )
    for jd, m in search.items():
        # blockchain-solidity is meant to qualify nobody, so the trap-count and
        # reading-size bands do not apply — it holds no traps and an empty result is the
        # point.
        #
        # **That the match is zero is asserted separately below.** While this posting was
        # exempt from every assertion, nothing guarded the zero. One "Solidity" written by
        # the prose layer would have created a qualified candidate and passed quietly.
        exempt = jd == "blockchain-solidity"
        lo, hi = SEARCH_STAGE_BAND
        in_band = lo <= m["search_stage"] <= hi
        mark = "  " if (exempt or in_band) else "!!"
        misplaced = str(m["misplaced"]) if m["misplaced"] else "-"
        print(
            f"  {mark} {jd:22} {m['agent_reads']:>6} {m['matched']:>6} "
            f"{m['search_stage']:>8} {m['traps']:>8} {m['density']:>7.1%}  {misplaced}"
        )
        if exempt and m["matched"]:
            violations.append(
                f"{jd}: {m['matched']} matched — this posting is meant to qualify "
                f"nobody. Someone now carries a Solidity token"
            )
        if not exempt and not in_band:
            violations.append(
                f"{jd}: {m['search_stage']} search-stage traps — the target is {lo}-{hi}. "
                f"{m['matched']} matched"
            )
        # The reading size binds `backend-rust` alone. That posting tests ranking quality,
        # which is what needs a readable set. The other two test emitting fewer than k and
        # choosing the mail's language, and both hold at 30 candidates.
        #
        # The distinction matters because the two Rust postings split one population;
        # requiring this size of all three would need an unrealistic number of Rust people.
        if jd == "backend-rust" and not read_lo <= m["agent_reads"] <= read_hi:
            violations.append(
                f"{jd}: the agent's query yields {m['agent_reads']} people, outside "
                f"{read_lo}-{read_hi}"
            )
        if m["misplaced"]:
            violations.append(
                f"{jd}: {m['misplaced']} traps placed on another posting appear here"
            )

    if violations:
        print(f"\n{len(violations)} violations:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    print("\nno violations")


if __name__ == "__main__":
    main()
