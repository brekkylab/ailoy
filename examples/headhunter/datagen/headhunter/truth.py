"""The answer key: facts about people, not rankings against a posting.

"Jane ranks first" presumes one posting and says nothing about another. "Jane's real
tenure is 89 months and the naive sum is 96" holds whatever the posting, and an agent
calling that eight years is wrong either way. Per-posting expectations live in
`eval/expected/`.
"""

from pathlib import Path

from common.dates import merge_spans, months, naive_sum
from common.writer import dump
from headhunter.gen import assemble
from headhunter.schema import Candidate, gap_months, spans_for

# `spans_for` is re-exported for `validate.py`. It is defined in `schema.py` — the
# property belongs to `Candidate`, and one definition keeps the tests honest.
__all__ = ["spans_for", "truth_for", "main"]


def _capped_at_last_update(
    candidate: Candidate, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Cuts spans at `last_updated_at`, dropping ones that start after it.

    `spans_for` fills a current role's end with `AS_OF`, so without the cut, time after
    the profile went quiet counts as verified.
    """
    year, month = (int(p) for p in candidate.last_updated_at.split("-")[:2])
    cutoff = months(year, month)
    return [(start, min(end, cutoff)) for start, end in spans if start < cutoff]


def truth_for(candidate: Candidate) -> dict:
    """One entry in `ground_truth.json`."""
    spans = spans_for(candidate)
    return {
        "id": candidate.id,
        "naive_months": naive_sum(spans),
        "real_months": merge_spans(spans),
        # Tenure counted only to the last update. `real_months` believes the current role
        # still holds; this does not, and the difference is what rests on that belief.
        #
        # **Not the answer to the `stale-profile` trap.** That trap's 48-vs-19 is one ML
        # platform position, and which position counts as "ML platform" is decided by the
        # posting — so it lives in `eval/expected/`, not here. Measured: 56 of 600 cross
        # the 48-month threshold this way, which is what makes a stale profile an ordinary
        # background property rather than a fingerprint of the trap.
        "verifiable_months": merge_spans(_capped_at_last_update(candidate, spans)),
        "job_function": candidate.job_function,
        "seniority": candidate.seniority,
        "skills_listed": sorted(s.name for s in candidate.skills),
        # Empty at this stage — descriptions do not exist until the prose layer runs, and
        # validate.py recomputes it then.
        "skills_backed_by_description": [],
        "has_contact": bool(candidate.contacts),
        "gap_months": gap_months(spans),
        "profile_language": candidate.profile_language,
        "open_to_work": candidate.open_to_work,
        # Raw values, not judgments — whether they contradict a posting is the posting's
        # question. None when the person is not open to work.
        "desired_location_type": (
            candidate.open_to_work_prefs[0].location_type
            if candidate.open_to_work_prefs
            else None
        ),
        "desired_start": (
            candidate.open_to_work_prefs[0].start_date
            if candidate.open_to_work_prefs
            else None
        ),
        "language_proficiencies": {l.name: l.proficiency for l in candidate.languages},
        "degree_fields": [e.field_of_study for e in candidate.educations],
        "certification_names": [c.name for c in candidate.certifications],
        "trap": getattr(candidate, "_trap", None),
        "control_for": getattr(candidate, "_control_for", None),
        "verdict": getattr(candidate, "_verdict", None),
        "pair_with": getattr(candidate, "_pair_with", None),
        # Which posting's search this trap is placed to catch. validate.py counts traps
        # per posting from this.
        "jd": getattr(candidate, "_jd", None),
        # **The only path from a trap to its prose prompt.** `_narrate_hint` is an
        # instance attribute, so `asdict()` misses it and it never reaches
        # candidates.json; narrate.py does not import fixtures. Drop this and per-trap
        # prompting disappears silently, in a run that happens once.
        "narrate_hint": getattr(candidate, "_narrate_hint", "") or None,
    }


def main() -> None:
    candidates = assemble()
    truth = [truth_for(c) for c in candidates]
    out = Path(__file__).parent.parent.parent / "data" / "ground_truth.json"
    dump(out, truth)
    print(f"wrote {out} ({len(truth)} entries)")


if __name__ == "__main__":
    main()
