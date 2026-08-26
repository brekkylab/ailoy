"""The generation rules for the four side tables, and their consistency.

These tables were empty because zero rows passed as a valid result. What is asserted here
is that values exist, not that the structure is right.
"""

from headhunter.gen import assemble
from headhunter.profile import educations_for

ROSTER = assemble()


def test_education_is_held_at_a_realistic_rate():
    with_edu = [c for c in ROSTER if c.educations]
    ratio = len(with_edu) / len(ROSTER)
    assert 0.70 <= ratio <= 0.90, f"education rate {ratio:.2f}"


def test_the_last_degree_finishes_before_the_first_job():
    for c in ROSTER:
        if not c.educations or not c.positions:
            continue
        last_end = max(e.end_year for e in c.educations)
        first_start = min(p.start_year for p in c.positions)
        assert last_end <= first_start, f"{c.id}: graduated {last_end} > started {first_start}"


def test_education_is_deterministic_per_person():
    target = ROSTER[0]
    assert educations_for(target) == educations_for(target)


from headhunter.profile import certifications_for, languages_for

_LANG_NAME = {"en": "English", "ko": "Korean", "ja": "Japanese"}


def test_certifications_are_held_at_a_realistic_rate():
    n = sum(1 for c in ROSTER if certifications_for(c))
    ratio = n / len(ROSTER)
    assert 0.20 <= ratio <= 0.40, f"certification rate {ratio:.2f}"


def test_the_profile_language_is_listed_and_at_working_level():
    strong = {"NATIVE_OR_BILINGUAL", "PROFESSIONAL_WORKING"}
    for c in ROSTER:
        langs = languages_for(c)
        if not langs:
            continue
        want = _LANG_NAME[c.profile_language]
        hit = [l for l in langs if l.name == want]
        assert hit, f"{c.id}: profile language {want} is not in the language list"
        assert hit[0].proficiency in strong, f"{c.id}: {want} is {hit[0].proficiency}"


def test_no_language_is_listed_twice():
    for c in ROSTER:
        names = [l.name for l in languages_for(c)]
        assert len(names) == len(set(names)), f"{c.id}: duplicate languages {names}"


from headhunter.profile import prefs_for


def test_nobody_not_looking_carries_a_desired_arrangement():
    for c in ROSTER:
        if not c.open_to_work:
            assert prefs_for(c) == [], f"{c.id}: not open to work but carries preferences"


def test_most_people_looking_carry_a_desired_arrangement():
    open_people = [c for c in ROSTER if c.open_to_work]
    n = sum(1 for c in open_people if prefs_for(c))
    ratio = n / len(open_people)
    assert 0.60 <= ratio <= 0.80, f"preference rate {ratio:.2f}"


def test_the_four_tables_are_populated_after_the_post_pass():
    assert any(c.educations for c in ROSTER)
    assert any(c.certifications for c in ROSTER)
    assert any(c.languages for c in ROSTER)
    assert any(c.open_to_work_prefs for c in ROSTER)


from headhunter.fixtures import TRAPS
from headhunter.profile import TRAP_ASSIGNMENTS


def test_the_new_traps_are_registered():
    added = {"prefs-mismatch", "language-overread",
              "degree-without-practice", "cert-without-practice"}
    assert added <= set(TRAPS)


def test_every_assigned_person_is_on_the_roster():
    ids = {c.id for c in ROSTER}
    for cid in TRAP_ASSIGNMENTS:
        assert cid in ids, f"{cid} is not on the roster"


def test_an_assignment_never_overwrites_an_existing_trap():
    by_id = {c.id: c for c in ROSTER}
    for cid, assigned in TRAP_ASSIGNMENTS.items():
        c = by_id[cid]
        held = getattr(c, "_trap", None)
        if held and held != assigned.get("trap"):
            raise AssertionError(f"{cid}: would overwrite the existing trap {held}")


def test_prefs_mismatch_wants_remote_only():
    for cid, a in TRAP_ASSIGNMENTS.items():
        if a.get("trap") != "prefs-mismatch":
            continue
        assert all(p.location_type == "Remote" for p in a["prefs"])


def test_assigned_values_reach_the_roster():
    by_id = {c.id: c for c in ROSTER}
    for cid, a in TRAP_ASSIGNMENTS.items():
        c = by_id[cid]
        if "prefs" in a:
            assert c.open_to_work_prefs == a["prefs"], f"{cid}: preferences were not applied"
        if "languages" in a:
            assert c.languages == a["languages"], f"{cid}: languages were not applied"
        if "educations" in a:
            assert c.educations == a["educations"], f"{cid}: education was not applied"
        if "certifications" in a:
            assert c.certifications == a["certifications"], f"{cid}: certifications were not applied"


from headhunter.fixtures import ASSIGNED_TRAPS


def test_each_assigned_trap_has_exactly_one_control():
    """The structural check over `core()` misses the assigned traps, so the same thing is
    asserted here. Without a control, a shallow rule can pass."""
    for name in ASSIGNED_TRAPS:
        traps = [a for a in TRAP_ASSIGNMENTS.values() if a.get("trap") == name]
        controls = [a for a in TRAP_ASSIGNMENTS.values() if a.get("control_for") == name]
        assert len(traps) == 1, f"{name}: {len(traps)} traps"
        assert len(controls) == 1, f"{name}: {len(controls)} controls"


def test_a_trap_and_its_control_answer_the_same_posting():
    by_id = {c.id: c for c in ROSTER}
    for name in ASSIGNED_TRAPS:
        jds = {a["jd"] for a in TRAP_ASSIGNMENTS.values()
               if name in (a.get("trap"), a.get("control_for"))}
        assert len(jds) == 1, f"{name}: postings split across {jds}"


def test_the_cert_trap_differs_from_its_control_only_in_practice():
    """Both hold a CKA. The only difference may be the evidence of practice."""
    pair = [a for a in TRAP_ASSIGNMENTS.values()
            if "cert-without-practice" in (a.get("trap"), a.get("control_for"))]
    names = [{c.name for c in a["certifications"]} for a in pair]
    assert names[0] == names[1], f"the certificates differ: {names}"


from headhunter.truth import truth_for


def test_the_answer_key_carries_the_side_table_facts():
    target = next(c for c in ROSTER if c.open_to_work_prefs and c.languages)
    t = truth_for(target)
    for key in ("desired_location_type", "desired_start",
                "language_proficiencies", "degree_fields", "certification_names"):
        assert key in t, f"{key} is missing from the answer key"
    assert t["desired_location_type"] == target.open_to_work_prefs[0].location_type


def _is_japanese(s: str) -> bool:
    return any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in s)


def test_tokyo_holds_japanese_profiles():
    ja = [c for c in ROSTER if c.profile_language == "ja"]
    assert 30 <= len(ja) <= 50, f"{len(ja)} Japanese profiles"
    assert all(c.country == "JP" for c in ja)


def test_a_japanese_profile_has_a_japanese_name_and_headline():
    for c in ROSTER:
        if c.profile_language != "ja":
            continue
        assert _is_japanese(c.first_name + c.last_name), f"{c.id}: named {c.first_name} {c.last_name}"
        assert _is_japanese(c.headline), f"{c.id}: headline is {c.headline}"


def test_nobody_japanized_is_a_trap_or_a_control():
    """`same-name` and `duplicate-profile` depend on the names. Renaming a hand-written
    person kills the trap silently."""
    for c in ROSTER:
        if c.profile_language != "ja":
            continue
        for field in ("_trap", "_control_for", "_verdict"):
            assert not getattr(c, field, None), f"{c.id}: someone with {field} was japanized"
        assert c.id not in TRAP_ASSIGNMENTS


def test_a_japanese_profile_lists_japanese():
    for c in ROSTER:
        if c.profile_language != "ja" or not c.languages:
            continue
        assert any(l.name == "Japanese" for l in c.languages), f"{c.id}: {[l.name for l in c.languages]}"


def _is_korean(s: str) -> bool:
    return any("가" <= ch <= "힣" for ch in s)


def test_a_japanese_profile_does_not_split_languages():
    """Changing only the headline leaves a Korean title inside a Japanese profile. Only the
    newly japanized people are checked; existing en and ko mismatches are left alone."""
    for c in ROSTER:
        if c.profile_language != "ja":
            continue
        for p in c.positions:
            assert not _is_korean(p.title), f"{c.id}: Korean title {p.title!r} in a Japanese profile"
        for n in c.contacts:
            assert _is_japanese(n.note), f"{c.id}: non-Japanese contact note {n.note!r}"


def test_some_qualified_tokyo_candidates_write_in_japanese():
    """For the Tokyo posting to test the language axis, some of the qualified have to write
    in Japanese. They were once all English, because every qualified person sat in the
    hand-written 65 that `japanize` leaves alone."""
    NEED = {"Kubernetes", "MLOps", "PyTorch"}
    qualified = [
        c for c in ROSTER
        if c.city == "Tokyo" and NEED <= {s.name for s in c.skills}
    ]
    ja = [c for c in qualified if c.profile_language == "ja"]
    assert len(qualified) >= 10, f"{len(qualified)} qualified in Tokyo"
    assert len(ja) >= 2, f"{len(ja)} of the qualified write in Japanese"


def test_a_granted_skill_uses_its_canonical_name():
    """`check_must_haves` compares with `s.name IN (...)`, so a variant would be in the
    data and not counted."""
    from headhunter.profile import TOKYO_SKILL_GRANTS
    by_id = {c.id: c for c in ROSTER}
    for cid, want in TOKYO_SKILL_GRANTS.items():
        names = {s.name for s in by_id[cid].skills}
        assert want in names, f"{cid}: {want} is not there under its canonical name — {sorted(names)}"
