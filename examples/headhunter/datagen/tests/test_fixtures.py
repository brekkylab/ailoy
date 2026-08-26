"""Whether the core 65's trap conditions still hold.

A failure here means a trap is dead — the data still looks plausible and there is nothing
left to catch. It happens silently, which is why a test is the only defense.
"""

import json
from collections import Counter

from common.dates import AS_OF, merge_spans, months, naive_sum
from headhunter.fixtures import ASSIGNED_TRAPS, STALE_BEFORE, TRAPS, core

# From `schema.py`, which `truth.py` also imports — one definition keeps what this test
# asserts and what reaches `ground_truth.json` from parting. Not from `truth.py` directly,
# which would drag the whole generator into a fixture test.
from headhunter.schema import gap_months, spans_for

# The same split FTS5's tokenizer does. A private name, but reimplementing it here would
# let "the search finds this" mean something different in the test than in the app.
from headhunter.variants import _tokens as fts5_tokens

spans_of = spans_for


def by_trap(name: str) -> list:
    return [c for c in core() if getattr(c, "_trap", None) == name]


def control_for(name: str):
    """The control for the `name` trap.

    Found by `_control_for` rather than by id: a hardcoded id would let the test quietly
    look at a different person, passing while guarding nothing.
    """
    controls = [c for c in core() if getattr(c, "_control_for", None) == name]
    assert len(controls) == 1, f"{name} has {len(controls)} controls"
    return controls[0]


def gap_of(candidate) -> int:
    """The longest gap between positions. See `schema.gap_months`."""
    return gap_months(spans_of(candidate))


def test_there_are_sixty_five_of_them():
    assert len(core()) == 65


def test_every_id_is_unique():
    ids = [c.id for c in core()]
    assert len(set(ids)) == len(ids)


def test_all_fourteen_trap_keys_are_present():
    """Thirteen trap kinds expressed as fourteen keys.

    Double-counted tenure splits in two: `overlapping-tenure` is one person's concurrent
    employment, `rank-inversion-pair` is two people whose order flips. Same arithmetic
    error, different places it shows — a filter and a ranking — so each needs its own data.
    """
    assert len(TRAPS) - len(ASSIGNED_TRAPS) == 14
    for name in TRAPS:
        if name in ASSIGNED_TRAPS:
            continue
        assert by_trap(name), f"trap {name!r} has no candidate"


def test_search_stage_traps_are_two_or_three_per_jd():
    """Two or three search-stage traps per posting.

    An absolute count rather than a share, because a share is derived: the same two people
    are 5% of 40 and 3.3% of 60. Controls are not traps and are not counted.
    """
    per_jd: dict[str, int] = {}
    for candidate in core():
        name = getattr(candidate, "_trap", None)
        if name and TRAPS[name].stage == "search":
            per_jd[candidate._jd] = per_jd.get(candidate._jd, 0) + 1

    assert per_jd, "no search-stage traps at all"
    for jd, count in sorted(per_jd.items()):
        assert 2 <= count <= 3, f"{jd} has {count} search-stage traps — the target is 2-3"


def test_every_search_stage_jd_is_covered():
    """All three postings carry search-stage traps. An empty one tests less."""
    covered = {
        candidate._jd
        for candidate in core()
        if getattr(candidate, "_trap", None) and TRAPS[candidate._trap].stage == "search"
    }
    assert covered == {"backend-rust", "ml-platform-tokyo", "backend-seoul-ko"}


def test_blockchain_jd_carries_no_traps():
    # Qualifying nobody is this posting's purpose; a trap here would blur it.
    assert not [s for s in TRAPS.values() if s.jd == "blockchain-solidity"]


def test_a_control_shares_its_traps_jd():
    """A control answering a different search is no comparison."""
    for candidate in core():
        target = getattr(candidate, "_control_for", None)
        if target:
            assert candidate._jd == TRAPS[target].jd, (
                f"{candidate.id} controls for {target} but sits on another posting"
            )


# Traps that need no separate control: the pair is the comparison.
SELF_CONTRASTING = frozenset({"rank-inversion-pair", "duplicate-profile", "same-name"})


def test_every_trap_has_a_control_unless_it_is_self_contrasting():
    """Without a control, a shallow rule can pass a trap.

    Three are exceptions. In `rank-inversion-pair`, A and B control for each other.

    `duplicate-profile` and `same-name` **control for each other across kinds.** Two
    profiles sharing a name are either one person, in which case only one belongs in the
    top k, or two, in which case both can. "Same name means duplicate" gets `same-name`
    wrong; "same name means different people" gets `duplicate-profile` wrong. Only
    comparing the companies and the dates gets both right.
    """
    for name in TRAPS:
        if name in SELF_CONTRASTING or name in ASSIGNED_TRAPS:
            continue
        controls = [c for c in core() if getattr(c, "_control_for", None) == name]
        assert len(controls) == 1, f"trap {name!r} has {len(controls)} controls"


def test_the_self_contrasting_traps_have_no_separate_control():
    for name in SELF_CONTRASTING:
        controls = [c for c in core() if getattr(c, "_control_for", None) == name]
        assert not controls, f"{name} is its own control but has {len(controls)}"


def test_the_core_headcount_breaks_down_as_spec_says():
    """65 = 17 traps + 11 controls + 37 judged.

    Seventeen because three kinds hold two people each; eleven because those same three
    are their own control.
    """
    roster = core()
    traps = [c for c in roster if getattr(c, "_trap", None)]
    controls = [c for c in roster if getattr(c, "_control_for", None)]
    verdicts = [c for c in roster if getattr(c, "_verdict", None)]

    assert len(traps) == 17, f"{len(traps)} trap people"
    assert len(controls) == 11, f"{len(controls)} controls"
    assert len(verdicts) == 37, f"{len(verdicts)} judged"
    assert len(traps) + len(controls) + len(verdicts) == 65


def test_the_overlapping_tenure_trap_actually_overlaps():
    trap = by_trap("overlapping-tenure")[0]
    spans = spans_of(trap)
    assert naive_sum(spans) > merge_spans(spans), "the spans do not overlap"


def test_its_control_does_not_overlap():
    control = [c for c in core() if getattr(c, "_control_for", None) == "overlapping-tenure"][0]
    spans = spans_of(control)
    assert naive_sum(spans) == merge_spans(spans), "the control's spans overlap"
    assert len(spans) >= 2, "the control needs more than one position to be a control"


def test_the_rank_inversion_pair_inverts():
    pair = by_trap("rank-inversion-pair")
    assert len(pair) == 2
    a, b = sorted(pair, key=lambda c: naive_sum(spans_of(c)), reverse=True)

    # a leads on the naive sum; b leads once merged.
    assert naive_sum(spans_of(a)) > naive_sum(spans_of(b))
    assert merge_spans(spans_of(a)) < merge_spans(spans_of(b))

    # Both clear the four-year must-have, so no filter separates them.
    assert merge_spans(spans_of(a)) >= 48
    assert merge_spans(spans_of(b)) >= 48

    # They point at each other.
    assert a._pair_with == b.id and b._pair_with == a.id


def test_the_skills_without_evidence_trap_has_no_evidence():
    trap = by_trap("skills-without-evidence")[0]
    assert any("rust" in s.name.lower() for s in trap.skills)
    # Descriptions are empty until the prose layer; `narrate_hint` carries the constraint.
    assert "no rust" in trap._narrate_hint.lower()


def test_the_inflated_title_trap_is_a_cxo_at_a_tiny_company():
    trap = by_trap("inflated-title")[0]
    assert trap.seniority == "CXO"
    assert trap.positions[0].company_size == "1-10"


def test_its_control_is_a_cxo_at_a_large_company():
    control = [c for c in core() if getattr(c, "_control_for", None) == "inflated-title"][0]
    assert control.seniority == "CXO"
    assert control.positions[0].company_size in ("1,001-5,000", "5,001-10,000", "10,001+")


def test_the_duplicate_profile_trap_is_two_people_who_are_one():
    pair = by_trap("duplicate-profile")
    assert len(pair) == 2
    assert pair[0].id != pair[1].id
    # The evidence they are one person: same company, same dates.
    assert pair[0].positions[0].company_urn == pair[1].positions[0].company_urn


def test_the_korean_only_profile_is_marked_korean():
    trap = by_trap("korean-only-profile")[0]
    assert trap.profile_language == "ko"


def test_the_no_contact_trap_has_no_contact_row():
    trap = by_trap("no-contact")[0]
    assert trap.contacts == []


def test_the_stale_profile_was_updated_long_ago():
    trap = by_trap("stale-profile")[0]
    assert trap.last_updated_at < STALE_BEFORE


# ─── what each control actually controls for ─────────────────────────────────
#
# Only two tests above look at an axis directly. The other nine controls were filtered by
# "same posting" and "exactly one", and neither asks **what is being controlled for**.
#
# That gap let a defect through: the `employment-gap` control's gap was 1 rather than 0
# because of the half-open interval, and pytest stayed green.
#
# Each test below asserts two things — **the axis differs** and **the rest matches**.
# Without the second, a control is just another person.
#
# Nine of eleven are here. The other two turn on `positions.description`, which the prose
# layer fills, so a test would pass whatever it asserted.


def test_the_not_open_control_is_open():
    trap = by_trap("strong-but-not-open")[0]
    control = control_for("strong-but-not-open")
    assert trap.open_to_work is False and control.open_to_work is True
    # Equal tenure is what makes "strong but not looking" the only difference.
    assert merge_spans(spans_of(trap)) == merge_spans(spans_of(control))


def test_the_shiny_control_has_the_must_have_skills():
    """Same title and company size as the trap, with the must-have actually done.

    This control makes "a senior title means no practice" wrong.
    """
    must_have = {"Kubernetes", "MLOps"}
    trap = by_trap("shiny-but-unqualified")[0]
    control = control_for("shiny-but-unqualified")
    assert must_have <= {s.name for s in control.skills}
    assert not must_have & {s.name for s in trap.skills}
    # Both are Director or above — the axis that is not being contrasted.
    senior_titles = ("Director", "VP", "CXO", "Owner")
    assert trap.seniority in senior_titles and control.seniority in senior_titles


def test_the_stale_control_was_updated_recently():
    control = control_for("stale-profile")
    assert control.last_updated_at >= "2025-01-01"
    # Equal spans are what leave the update date as the only difference; otherwise the
    # tenure itself changes rather than whether it is an estimate.
    trap = by_trap("stale-profile")[0]
    assert spans_of(trap) == spans_of(control)


def test_the_gap_control_has_no_gap():
    """The axis where a one-month gap actually survived. Half-open, it looked adjacent."""
    trap = by_trap("employment-gap")[0]
    control = control_for("employment-gap")
    assert gap_of(trap) >= 12, f"the trap's gap is {gap_of(trap)} months — it must be 12 or more"
    assert gap_of(control) == 0, f"the control's gap is {gap_of(control)} months — it must be 0"
    # Equal total tenure is what leaves the gap as the only difference.
    assert merge_spans(spans_of(trap)) == merge_spans(spans_of(control))


def test_the_location_control_is_in_korea():
    trap = by_trap("location-mismatch")[0]
    control = control_for("location-mismatch")
    assert trap.country != "KR" and control.country == "KR"
    # Identical skills are what leave the location as the only difference.
    assert {s.name for s in trap.skills} == {s.name for s in control.skills}


def test_the_korean_only_control_is_in_english():
    """The mail's language follows `profile_language`, not the name.

    Both have Korean names and work in Seoul. Only the profile's language differs.
    """
    trap = by_trap("korean-only-profile")[0]
    control = control_for("korean-only-profile")
    assert trap.profile_language == "ko" and control.profile_language == "en"
    assert trap.country == control.country == "KR"


def test_the_no_contact_control_has_a_contact_row():
    trap = by_trap("no-contact")[0]
    control = control_for("no-contact")
    assert trap.contacts == [] and control.contacts != []
    # Equal skills are what leave "strong but unreachable" as the only difference.
    assert {s.name for s in trap.skills} == {s.name for s in control.skills}


def test_industry_is_never_the_contrast_axis():
    """`industry` controls for no trap, so it has to match within a pair.

    The column is scattered to remove a fingerprint. If that scattering gives a trap and its
    control different values, the comparison has two axes and there is no telling which one
    the judgment turned on — each value correct, the contrast broken.

    The same-name pair is excluded: they are different people, and giving them different
    industries adds evidence of that.
    """
    for name in TRAPS:
        if name in SELF_CONTRASTING or name in ASSIGNED_TRAPS:
            continue
        trap, control = by_trap(name)[0], control_for(name)
        assert control.industry == trap.industry, (
            f"{name}: the trap is {trap.industry!r} and the control {control.industry!r} "
            "— the comparison now has two axes"
        )

    # A duplicate profile is **the same person**, so it has to match.
    first, second = by_trap("duplicate-profile")
    assert first.industry == second.industry


def test_the_inflated_title_pair_differs_only_in_company_size():
    """The title must not distinguish them. That is the whole trap.

    Two tests each look at one person. Both can pass while the two differ in function or
    city as well, and then "same title, different size" is not the comparison being made.
    """
    trap = by_trap("inflated-title")[0]
    control = control_for("inflated-title")
    assert trap.seniority == control.seniority == "CXO"
    assert trap.positions[0].company_size == "1-10"
    assert control.positions[0].company_size in ("1,001-5,000", "5,001-10,000", "10,001+")
    # They have to answer the same search to be seen side by side.
    assert trap.job_function == control.job_function
    assert trap.country == control.country


def test_the_overlapping_tenure_pair_both_clear_the_must_have():
    """Two positions each, differing only in the overlap — and **both clear the filter.**

    Otherwise the trap is removed before ranking. The trap's real tenure is exactly 48
    months: summed naively it reads as six years and ranks up; counted correctly it barely
    meets the minimum.
    """
    trap = by_trap("overlapping-tenure")[0]
    control = control_for("overlapping-tenure")
    assert len(trap.positions) == len(control.positions) == 2
    assert naive_sum(spans_of(trap)) > merge_spans(spans_of(trap))
    assert naive_sum(spans_of(control)) == merge_spans(spans_of(control))
    assert merge_spans(spans_of(trap)) >= 48
    assert merge_spans(spans_of(control)) >= 48


def test_the_clear_fits_are_five():
    assert len([c for c in core() if getattr(c, "_verdict", None) == "clear-fit"]) == 5


def test_the_clear_misses_are_twenty():
    assert len([c for c in core() if getattr(c, "_verdict", None) == "clear-miss"]) == 20


def test_the_borderlines_are_twelve():
    assert len([c for c in core() if getattr(c, "_verdict", None) == "borderline"]) == 12


# ─── whether the answer leaks, and whether the evidence is unique ─────────────
#
# These three were run by hand. One of them caught the one-month defect, and because it
# was not in the repository the defect reached a commit — running a check is not the same
# as keeping it.


def test_the_private_fields_never_reach_the_json():
    """Whether the answer key leaks into the profiles.

    **The worst leak available.** `_trap` or `_verdict` reaching `candidates.json` puts the
    answer in what the agent reads and makes every fingerprint precaution pointless.

    Scanned by prefix rather than by name: naming the fields means forgetting to add the
    next one, and what is forgotten leaks quietly.
    """
    for candidate in core():
        private = {k: v for k, v in vars(candidate).items() if k.startswith("_")}
        assert private, f"{candidate.id}: no private fields at all"
        blob = json.dumps(candidate.to_json(), ensure_ascii=False, sort_keys=True)
        for name, value in private.items():
            assert name not in blob, f"{candidate.id}: {name!r} is in to_json()"
            if isinstance(value, str) and value:
                assert value not in blob, (
                    f"{candidate.id}: {name}'s value {value[:40]!r} is in to_json()"
                )


def test_the_only_repeated_names_are_the_two_pairs_that_need_them():
    """A shared name is what two traps rest on. A third one blurs the evidence.

    Two profiles sharing a name split into two cases — `duplicate-profile` and
    `same-name` — and each is the other's control. An accidental third puts a case in front
    of the agent for which no answer is defined.

    Derived from `_trap` rather than written out, so it follows a change of person.
    """
    counts = Counter((c.first_name, c.last_name) for c in core())
    repeated = {name for name, n in counts.items() if n > 1}
    intended = {
        (c.first_name, c.last_name)
        for c in core()
        if getattr(c, "_trap", None) in ("duplicate-profile", "same-name")
    }
    assert repeated == intended, (
        f"의도하지 않은 동명: {sorted(repeated - intended)}, "
        f"이름이 갈라진 쌍: {sorted(intended - repeated)}"
    )
    for name in intended:
        assert counts[name] == 2, f"{name} 이 {counts[name]}명이다 — 쌍이어야 한다"

    # Whether the two pairs really are two cases. The point is that the name cannot decide,
    # so without the company separating them the two traps collapse into one.
    duplicate = by_trap("duplicate-profile")
    same_name = by_trap("same-name")
    assert duplicate[0].positions[0].company_urn == duplicate[1].positions[0].company_urn
    assert same_name[0].positions[0].company_urn != same_name[1].positions[0].company_urn


def test_only_the_duplicate_profile_shares_a_company_and_a_period():
    """Same company, same dates is the only evidence of one person. Two such pairs and it
    is no longer evidence.

    Company names are combinations and the dates are hand-written, so two unrelated people
    can land on the same pair by accident. The duplicate-profile trap's evidence would then
    apply to them too, and whether they should be merged has no answer.
    """
    seen: dict[tuple, list[str]] = {}
    for candidate in core():
        for position in candidate.positions:
            key = (
                position.company_urn,
                position.start_year,
                position.start_month,
                position.end_year,
                position.end_month,
            )
            seen.setdefault(key, []).append(candidate.id)

    collisions = {key: ids for key, ids in seen.items() if len(ids) > 1}
    assert len(collisions) == 1, f"같은 회사·같은 기간이 {len(collisions)}쌍이다: {collisions}"
    assert sorted(next(iter(collisions.values()))) == sorted(
        c.id for c in by_trap("duplicate-profile")
    )


def test_every_candidate_says_which_jd_brings_it_up():
    """A missing `_jd` drops that person from the per-posting counts, silently.

    `truth.py` writes it into `ground_truth.json` and `validate.py` counts traps per
    posting from it. `None` raises nothing — a trap simply stops being counted, and the
    per-posting target reads as missed or met for the wrong reason.
    """
    orphans = [c.id for c in core() if not getattr(c, "_jd", None)]
    assert not orphans, f"_jd 가 없는 후보: {orphans}"


# Who may carry no hint, and why.
#
# A list rather than a count, so that a missing hint and a deliberately empty one are
# distinguishable. A count makes them identical, and the next person deleting a hint cannot
# tell which they did.
HINT_EXEMPT: dict[str, str] = {
    "urn:li:person:h7cvx2pn": (
        "B of the inversion pair. This person works through arithmetic alone — naive 89 "
        "equals merged 89, so there is no overlap to hide. A has the overlap and A has a "
        "hint."
    ),
}


def test_every_candidate_carries_a_hint_unless_it_is_exempt():
    """An empty `narrate_hint` means the constraint never reaches the prose layer.

    This field is the only path from a trap to its prompt. Empty, the LLM writes with no
    constraint at all.

    **That matters most for the twenty clear misses.** One line of "I have done a little
    Rust too" in an eight-year frontend profile ends the miss being clear, it happens
    without an error, and the prose layer runs once.
    """
    roster = core()
    ids = {c.id for c in roster}
    for ident, reason in HINT_EXEMPT.items():
        # No stale exemptions: if the person is gone, the exemption goes too.
        assert ident in ids, f"the exempt {ident} is not on the roster — a stale exemption"
        assert reason.strip(), f"{ident} is exempt with no reason given"

    missing = {c.id for c in roster if not getattr(c, "_narrate_hint", "")}
    assert missing <= set(HINT_EXEMPT), (
        f"candidates with no hint and no reason: {sorted(missing - set(HINT_EXEMPT))}"
    )

    # A clear miss cannot be exempt. The hint is the only defense for these twenty, so the
    # exemption mechanism has to be closed to them.
    for candidate in roster:
        if getattr(candidate, "_verdict", None) == "clear-miss":
            assert candidate.id not in HINT_EXEMPT, f"{candidate.id}: a clear miss cannot be exempt"
            assert candidate._narrate_hint, f"{candidate.id}: a clear miss with no hint"


def test_the_headline_bait_trap_carries_rust_only_in_its_headline():
    """This trap is defined in the structure layer alone, so it is checkable now.

    Its control needs the prose layer, but the trap is entirely "in the headline and nowhere
    else".

    That is also **the single-surface risk**: the only path by which this person is found is
    the headline, so an index that omits that column removes the trap.
    """
    trap = by_trap("headline-bait")[0]
    assert "rust" in fts5_tokens(trap.headline)
    assert not any("rust" in fts5_tokens(s.name) for s in trap.skills)
    assert not any("rust" in fts5_tokens(p.title) for p in trap.positions)


def test_the_skills_without_evidence_trap_carries_rust_only_in_its_skills():
    """The mirror of the trap above: in the skills, and not in the headline or the titles.

    The other test asks whether Rust is in the skills. This asks the other half — that it is
    **absent from the other surfaces** — without which this trap is indistinguishable from
    `headline-bait`.
    """
    trap = by_trap("skills-without-evidence")[0]
    assert any("rust" in fts5_tokens(s.name) for s in trap.skills)
    assert "rust" not in fts5_tokens(trap.headline)
    assert not any("rust" in fts5_tokens(p.title) for p in trap.positions)


def test_the_stale_profiles_tenure_hangs_on_believing_it_is_current():
    """The trap's teeth: believing the open-ended role decides whether the must-have is met.

    Believed, the position is 48 months and clears four years; counted to the last update it
    is 19 and does not. That difference is why the trap demands the tenure be called an
    estimate — without saying so, four years is asserted.

    Move the start a few months and both values land on the same side, turning the trap into
    an ordinary candidate. Silently.
    """
    trap = by_trap("stale-profile")[0]
    current = [p for p in trap.positions if p.end_year is None]
    assert len(current) == 1, "the arithmetic needs exactly one current position"

    start = months(current[0].start_year, current[0].start_month)
    year, month = (int(part) for part in trap.last_updated_at.split("-")[:2])
    believed = months(*AS_OF) - start
    verifiable = months(year, month) - start
    assert believed >= 48, f"{believed} months even when believed — it has to clear"
    assert verifiable < 48, f"{verifiable} months to the last update — it has to fall short"
    # Straddling 48 is not enough. Both assertions hold with a one-month unverified span,
    # and then there is nothing to disclose. The estimated stretch has to be substantial.
    assert believed - verifiable >= 12, (
        f"{believed - verifiable} months unverified — nothing to call an estimate"
    )


def test_the_months_the_spec_computed_are_still_those_months():
    """The four figures the two pairs were designed around. **All four are pinned.**

    The tests above check relations — does the order invert, do the spans overlap — and a
    relation cannot tell that the dates left the design. These two pairs of dates are the
    only property in this dataset that has to be designed by hand.

    **`AS_OF` is not a free knob.** B's second position is current, so moving `AS_OF` seven
    months makes B's naive sum 96, equal to A's, and **the inversion is gone**. A failure
    here is not a false alarm; it is the instruction to rederive the figures.

    The composition is asserted too. Give B an end date and `(89, 89)` still holds, so the
    numbers alone would miss B quietly ceasing to be current.
    """
    pair = by_trap("rank-inversion-pair")
    open_ended = [c for c in pair if any(p.end_year is None for p in c.positions)]
    closed = [c for c in pair if all(p.end_year is not None for p in c.positions)]
    assert len(open_ended) == 1 and len(closed) == 1, (
        "the pair has to be one current (B) and one ended (A) — the figures below assume "
        f"B's second position is open-ended. {len(open_ended)} current"
    )

    # A — the overlap is fully contained, so merging drops it to 60. Independent of AS_OF.
    got_a = (naive_sum(spans_of(closed[0])), merge_spans(spans_of(closed[0])))
    assert got_a == (96, 60), f"A: {got_a} — the design says (96, 60)"
    # B — no overlap, so the two agree. Relative to AS_OF, as the docstring explains.
    got_b = (naive_sum(spans_of(open_ended[0])), merge_spans(spans_of(open_ended[0])))
    assert got_b == (89, 89), (
        f"B: {got_b} — the design says (89, 89). If AS_OF moved, rederive the figures and "
        "change this with them"
    )

    # The overlapping-tenure pair. Both are in the past, so independent of AS_OF.
    for label, candidate, expected in (
        ("the overlap trap", by_trap("overlapping-tenure")[0], (72, 48)),
        ("its control", control_for("overlapping-tenure"), (94, 94)),
    ):
        assert all(p.end_year is not None for p in candidate.positions), (
            f"{label} gained a current position — these figures now depend on AS_OF and "
            "have to be rederived"
        )
        got = (naive_sum(spans_of(candidate)), merge_spans(spans_of(candidate)))
        assert got == expected, f"{label}: {got} — the design says {expected}"
