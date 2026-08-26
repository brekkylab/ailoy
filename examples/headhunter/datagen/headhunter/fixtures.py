"""The core 65: 14 trap kinds (17 people), their 11 controls, and 37 for judgment.

**Written by hand.** A rule can produce "someone whose spans overlap", but not "two people
whose order flips between the naive sum and the merge" — that comes from designing two
sets of dates together.

`_trap`, `_control_for`, `_verdict`, `_pair_with`, `_jd`, and `_narrate_hint` are not
`Candidate` fields and never reach `to_json()`. `truth.py` reads them into
`ground_truth.json`, so the answer key stays out of the profiles.

**The vocabulary is shared with the background.** `job_function`, skill names, and cities
come from the same pools `gen.py` draws on. A value only the core holds would let one
column separate all 65, and the agent could find them without reading anyone. Wanting a
skill that is not here means adding it to `gen.py`'s pool.

For the same reason `Solidity` appears nowhere: the blockchain posting is meant to qualify
nobody, and one token would end that. The five placed against it look adjacent through
their headlines only.

`_jd` says which posting's search should surface this person.

- `backend-rust` — Rust on an **English** profile
- `backend-seoul-ko` — Rust on a **Korean** profile. Same skills, different language
- `ml-platform-tokyo` — PyTorch and Kubernetes in Tokyo, and no Rust. Rust would put them
  in the `backend-rust` results too, and the trap would land on the wrong posting
"""

from dataclasses import dataclass

from common.names import urn_for
from headhunter.schema import Candidate, Contact, Position, Skill


@dataclass(frozen=True)
class TrapSpec:
    """One trap.

    `checks` states what has to hold for the trap to be alive. Nothing reads it
    mechanically; it is prose for whoever changes the data.

    `stage` says where the trap works. A `"search"` trap is one the search should not
    surface and does, so it has to be among the results; a `"judge"` trap is found
    correctly and misread afterwards, so it has to be near the top k.

    A control carries the **same `jd`** as its trap. Answering a different search makes it
    no comparison at all.

    `headcount` exists because `validate.py` counts from it. When the number lived in a
    comment, one half of a paired trap could lose its label and the kind count stayed the
    same, so it passed quietly.
    """

    name: str
    what_goes_wrong: str
    checks: str
    stage: str  # "search" | "judge"
    jd: str  # "backend-rust" | "ml-platform-tokyo" | "backend-seoul-ko"
    headcount: int = 1


# Three places use these thresholds: the `checks` prose below, `validate.py`'s trap
# conditions, and the fixture tests. Written out three times, moving one calls a person a
# trap who is not, or the reverse.
STALE_BEFORE = "2025-01-01"
GAP_MONTHS = 12

TRAPS: dict[str, TrapSpec] = {
    "headline-bait": TrapSpec(
        "headline-bait",
        "summary 에 관심만 있고 실제 경력은 없다. 키워드 매칭이면 상위로 올라온다",
        "positions 의 어느 description 에도 해당 기술이 없어야 한다",
        "search",
        "backend-rust",
    ),
    "skills-without-evidence": TrapSpec(
        "skills-without-evidence",
        "skills 에 Rust 가 있지만 positions.description 에 흔적이 없다",
        "skills 에 있고 description 에 없어야 한다",
        "search",
        "backend-rust",
    ),
    "overlapping-tenure": TrapSpec(
        "overlapping-tenure",
        "A consulting role overlaps a full-time one. Summed, the tenure is inflated",
        "naive_sum > merge_spans",
        "judge",
        "backend-rust",
    ),
    "rank-inversion-pair": TrapSpec(
        "rank-inversion-pair",
        "Two people whose order flips between the naive sum and the merge",
        "naive_sum and merge_spans order them oppositely, and both clear 48 months",
        "judge",
        "backend-rust",
        headcount=2,
    ),
    "duplicate-profile": TrapSpec(
        "duplicate-profile",
        "Two ids, one person. Both in the top k is a failure",
        "Two profiles sharing a company_urn and the same dates",
        "search",
        "ml-platform-tokyo",
        headcount=2,
    ),
    "shiny-but-unqualified": TrapSpec(
        "shiny-but-unqualified",
        "A known company and a senior title, with none of the must-have experience",
        # They do hold the buzzword skill (PyTorch) — that is why the search finds them.
        # What is missing is evidence of running a platform: Kubernetes and MLOps.
        "seniority is Director or above, PyTorch is present, Kubernetes and MLOps are not",
        "search",
        "ml-platform-tokyo",
    ),
    "location-mismatch": TrapSpec(
        "location-mismatch",
        "The right skills on another continent. Has to be recorded as a risk",
        "Every skill met, and `country` is not where this trap's posting is. For "
        "backend-seoul-ko that means not KR — the check is relative to the posting",
        "judge",
        "backend-seoul-ko",
    ),
    "strong-but-not-open": TrapSpec(
        "strong-but-not-open",
        "A strong fit who is not looking. Excluding or including-with-a-note is what is observed",
        "open_to_work is False and every must-have is met",
        "judge",
        "backend-rust",
    ),
    "stale-profile": TrapSpec(
        "stale-profile",
        "Two years without an update. That the tenure is an estimate has to be said",
        "last_updated_at < 2025-01-01",
        "judge",
        "ml-platform-tokyo",
    ),
    "employment-gap": TrapSpec(
        "employment-gap",
        "An 18-month gap. Recorded as a fact, with no invented reason",
        "Twelve months or more between two positions",
        "judge",
        "ml-platform-tokyo",
    ),
    "korean-only-profile": TrapSpec(
        "korean-only-profile",
        "A Korean-only profile. Forces a decision about the mail's language",
        "profile_language == 'ko'",
        "judge",
        "backend-seoul-ko",
    ),
    "same-name": TrapSpec(
        "same-name",
        "One name, two people, two companies",
        "Two profiles with the same name and different company_urns",
        "search",
        "backend-seoul-ko",
        headcount=2,
    ),
    "no-contact": TrapSpec(
        "no-contact",
        "One of the top k. Has to reach the list of things a person checks",
        "contacts is empty",
        "judge",
        "backend-seoul-ko",
    ),
    "inflated-title": TrapSpec(
        "inflated-title",
        "CTO of a three-person startup. The title alone is not the judgment",
        "seniority == 'CXO' and company_size == '1-10'",
        "judge",
        "ml-platform-tokyo",
    ),
    # The four below stand on the side tables alone, so they are assigned to background
    # people rather than written here. `profile.py` holds the assignments.
    "prefs-mismatch": TrapSpec(
        "prefs-mismatch",
        "Looking, but the desired arrangement contradicts the posting. Read the preference, not the flag",
        "open_to_work is True and open_to_work_prefs.location_type is Remote",
        "judge",
        "backend-rust",
    ),
    "language-overread": TrapSpec(
        "language-overread",
        "English listed as native while the profile is Korean. The mail still follows the profile",
        "profile_language is ko and languages holds English/NATIVE_OR_BILINGUAL",
        "judge",
        "backend-seoul-ko",
    ),
    "degree-without-practice": TrapSpec(
        "degree-without-practice",
        "A master's in machine learning with no Kubernetes operations. The posting wants all three",
        "educations holds a Machine Learning master's and no position shows Kubernetes practice",
        "judge",
        "ml-platform-tokyo",
    ),
    "cert-without-practice": TrapSpec(
        "cert-without-practice",
        "A CKA with no Kubernetes operations behind it. A certificate looks verified",
        "certifications holds a Certified Kubernetes Administrator and no position shows the practice",
        "judge",
        "ml-platform-tokyo",
    ),
}

# Those four are not built by `core()`. Checks that count the core 65 have to exclude
# them; what those checks guard is the structure of the hand-written people.
ASSIGNED_TRAPS: frozenset[str] = frozenset(
    {"prefs-mismatch", "language-overread", "degree-without-practice", "cert-without-practice"}
)


# **Nothing here may fingerprint the core 65.**
#
# A column carrying the same value for all 65 lets the agent find them by that column
# rather than by reading. `industry`, `connections_count`, and `last_updated_at` were all
# like this once, and two of them were not even parameters.
#
# **`ident` must not leak the answer.** Both `id` and `public_profile_url` are built from
# it, so a name like `rank-inv-a` would expose the trap through
# `WHERE id LIKE '%rank-inv%'`. Opaque eight characters instead.
#
# `validate.py` catches this systematically: no single column separates the core exactly.
def _candidate(
    ident: str,
    first: str,
    last: str,
    *,
    headline: str,
    seniority: str,
    city: str = "Seoul",
    country: str = "KR",
    language: str = "en",
    job_function: str = "Backend",
    industry: str = "Computer Software",
    open_to_work: bool = True,
    updated: str = "2026-06-01",
    connections: int = 500,
    positions: list[Position],
    skills: list[Skill],
    contacts: list[Contact] | None = None,
    trap: str | None = None,
    control_for: str | None = None,
    verdict: str | None = None,
    pair_with: str | None = None,
    jd: str | None = None,
    narrate_hint: str = "",
) -> Candidate:
    """One core person, with the private fields attached."""
    candidate = Candidate(
        id=f"urn:li:person:{ident}",
        first_name=first,
        last_name=last,
        headline=headline,
        city=city,
        country=country,
        industry=industry,
        # From gen.py's JOB_FUNCTIONS. A fixed value like "Engineering" would separate all
        # 65 in one column. Defaults to Backend; other functions are given at the call site.
        job_function=job_function,
        seniority=seniority,
        profile_language=language,
        open_to_work=open_to_work,
        # A parameter for the same reason: 500 across all 65 would be a fingerprint.
        connections_count=connections,
        last_updated_at=updated,
        public_profile_url=f"https://www.linkedin.com/in/{ident}",
        positions=positions,
        skills=skills,
        contacts=contacts or [],
    )
    candidate._trap = trap
    candidate._control_for = control_for
    candidate._verdict = verdict
    candidate._pair_with = pair_with
    candidate._jd = jd
    candidate._narrate_hint = narrate_hint
    return candidate


def _position(
    title: str,
    company: str,
    *,
    size: str,
    start: tuple[int, int],
    end: tuple[int, int] | None = None,
    employment: str = "FULL_TIME",
    workplace: str = "On-site",
    location: str = "Seoul, KR",
) -> Position:
    return Position(
        title=title,
        company_name=company,
        company_urn=urn_for(company),
        company_size=size,
        employment_type=employment,
        workplace_type=workplace,
        location=location,
        description="",  # filled by the prose layer
        start_year=start[0],
        start_month=start[1],
        end_year=end[0] if end else None,
        end_month=end[1] if end else None,
    )


# Matched to the profile language, or the prose layer cannot tell which to write in.
_INMAIL_EN = Contact("inmail", "InMail open")
_INMAIL_KO = Contact("inmail", "InMail 수신 가능")
_REFERRAL_EN = Contact("referral", "Referred by a former colleague")
_REFERRAL_KO = Contact("referral", "전 동료 소개 가능")

TOKYO = "Tokyo, JP"
BERLIN = "Berlin, DE"
SF = "San Francisco, US"


def _rank_inversion_pair() -> list[Candidate]:
    """The inversion pair. Change the dates and the inversion stops.

    **`AS_OF` is not a free knob.** B's second position is current, so moving it seven
    months makes B's naive sum 96 as well and the inversion collapses.

    `tests/test_fixtures.py` pins all four figures — A 96/60, B 89/89, and the overlapping
    pair's 72/48 and 94/94 — so a failure there is the instruction to rederive them.
    """
    # Opaque idents. `id` and `public_profile_url` are built from these.
    a_id, b_id = "t4qm9wbd", "h7cvx2pn"
    a = _candidate(
        a_id, "Morgan", "Thorne",
        headline="Backend Engineer · Rust, payments",
        seniority="Senior",
        positions=[
            # full-time, 60 months
            _position("Backend Engineer", "Peatmoor Systems", size="201-500",
                      start=(2018, 3), end=(2023, 3)),
            # consulting, 36 months, entirely inside the span above
            _position("Backend Consultant", "Larkfield Labs", size="11-50",
                      start=(2019, 6), end=(2022, 6), employment="CONTRACT"),
        ],
        skills=[Skill("Rust", 20), Skill("Distributed Systems", 12)],
        trap="rank-inversion-pair",
        pair_with=f"urn:li:person:{b_id}",
        jd="backend-rust",
        narrate_hint="Do not say the two spans overlap. Describe each position on its own.",
    )
    b = _candidate(
        b_id, "Avery", "Kendrick",
        headline="Senior Backend Engineer · Rust",
        seniority="Senior",
        positions=[
            _position("Backend Engineer", "Glasswing Works", size="51-200",
                      start=(2019, 1), end=(2021, 7)),
            _position("Senior Backend Engineer", "Ironvale Networks", size="501-1,000",
                      start=(2021, 9)),  # current
        ],
        skills=[Skill("Rust", 25), Skill("Kubernetes", 15)],
        trap="rank-inversion-pair",
        pair_with=f"urn:li:person:{a_id}",
        jd="backend-rust",
    )
    return [a, b]


def _traps() -> list[Candidate]:
    """The 17 trap people. The inversion pair comes from `_rank_inversion_pair()`.

    Grouped by posting because a search-stage trap targets two or three per posting: two
    for backend-rust (headline bait, skills without evidence), three for
    ml-platform-tokyo (two duplicate profiles and one shiny-but-unqualified), and two for
    backend-seoul-ko (the same-name pair).
    """
    return [
        # ─── backend-rust ────────────────────────────────────────────────
        # Rust is in the headline and not in the skills — that is what separates this from
        # skills-without-evidence. The headline is indexed, so the search finds them.
        _candidate(
            "q7mv2xkd", "Casey", "Ashby",
            headline="Backend Engineer · Python, PostgreSQL · learning Rust",
            seniority="Senior",
            updated="2026-05-12",
            connections=842,
            positions=[
                _position("Backend Engineer", "Mossbank Systems", size="201-500",
                          start=(2020, 3)),
                _position("Software Engineer", "Draftwell Labs", size="51-200",
                          start=(2017, 6), end=(2020, 2)),
            ],
            skills=[Skill("Python", 34), Skill("PostgreSQL", 21), Skill("Java", 12)],
            trap="headline-bait",
            jd="backend-rust",
            narrate_hint="Write Rust as something they want to learn, never as work done — no Rust work in any "
            "position description.",
        ),
        # Rust in the skills and in no position. `_narrate_hint` carries that constraint to
        # the prose layer; descriptions are empty here, so it cannot be checked yet.
        _candidate(
            "h4tb9wzr", "Jordan", "Merrick",
            headline="Senior Backend Engineer · Java, Kafka",
            seniority="Senior",
            updated="2026-07-03",
            connections=1204,
            positions=[
                _position("Senior Backend Engineer", "Foldgate Networks", size="501-1,000",
                          start=(2021, 1)),
                _position("Backend Engineer", "Halcyon Works", size="51-200",
                          start=(2018, 4), end=(2020, 12)),
            ],
            skills=[Skill("Rust", 7), Skill("Java", 41), Skill("Kafka", 24)],
            contacts=[_INMAIL_EN],
            trap="skills-without-evidence",
            jd="backend-rust",
            narrate_hint="Rust is in the skills, but both positions describe Java and Kafka work — no Rust in "
            "any position description.",
        ),
        # Naive 72 months, merged 48 — exactly the four-year must-have. Summing reads this
        # person as six years and ranks them up. The control is 94/94.
        _candidate(
            "v2ncq8jf", "Riley", "Calloway",
            headline="Backend Engineer · Rust, distributed systems",
            seniority="Senior",
            updated="2026-04-18",
            connections=690,
            positions=[
                _position("Senior Backend Engineer", "Emberpath Technologies",
                          size="201-500", start=(2019, 6), end=(2023, 6)),
                _position("Backend Consultant", "Nightjar Labs", size="11-50",
                          start=(2020, 9), end=(2022, 9), employment="CONTRACT"),
            ],
            skills=[Skill("Rust", 19), Skill("Distributed Systems", 11)],
            contacts=[_INMAIL_EN],
            trap="overlapping-tenure",
            jd="backend-rust",
            narrate_hint="Do not say the two spans overlap. Describe each position on its own.",
        ),
        # A strong fit who is not looking. No signal beyond the flag itself.
        _candidate(
            "d8kf3prw", "Rowan", "Voss",
            headline="Staff Backend Engineer · Rust, distributed systems",
            seniority="Senior",
            open_to_work=False,
            updated="2026-02-09",
            connections=1571,
            positions=[
                _position("Staff Backend Engineer", "Kelpstone Dynamics",
                          size="1,001-5,000", start=(2022, 2)),
                _position("Backend Engineer", "Arborline Works", size="201-500",
                          start=(2018, 1), end=(2022, 1)),
            ],
            skills=[Skill("Rust", 33), Skill("Distributed Systems", 22),
                    Skill("Kubernetes", 12)],
            contacts=[_REFERRAL_EN],
            trap="strong-but-not-open",
            jd="backend-rust",
            narrate_hint="Do not say they are not looking — the open_to_work flag carries that alone.",
        ),
        # ─── ml-platform-tokyo ───────────────────────────────────────────
        # Two profiles of one person. Same company and dates is the only evidence; the name
        # spelling drifts (`Inc.`) but the urn does not. A shared name proves nothing —
        # `same-name` is the trap that makes that rule wrong.
        _candidate(
            "b6xr4tqm", "Kai", "Lockhart",
            headline="ML Platform Engineer · Kubernetes, PyTorch",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            industry="Internet",
            updated="2026-07-11",
            connections=913,
            positions=[
                _position("ML Platform Engineer", "Quantile Labs", size="501-1,000",
                          start=(2021, 5), location=TOKYO),
                _position("Machine Learning Engineer", "Junipex Platform", size="201-500",
                          start=(2018, 8), end=(2021, 4), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 27), Skill("Python", 33), Skill("MLOps", 18),
                    Skill("PyTorch", 14)],
            contacts=[_INMAIL_EN],
            trap="duplicate-profile",
            pair_with="urn:li:person:m9jd7vhs",
            jd="ml-platform-tokyo",
            narrate_hint="The same person as another profile, but never say so. Write the same company and "
            "dates in this profile's own words.",
        ),
        _candidate(
            "m9jd7vhs", "Kai", "Lockhart",
            headline="MLOps Engineer · ML infrastructure, PyTorch",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            industry="Internet",
            updated="2025-10-22",
            connections=388,
            positions=[
                # Same company, same start. Only the spelling differs.
                _position("ML Platform Engineer", "Quantile Labs Inc.", size="501-1,000",
                          start=(2021, 5), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 9), Skill("Python", 15), Skill("PyTorch", 6)],
            trap="duplicate-profile",
            pair_with="urn:li:person:b6xr4tqm",
            jd="ml-platform-tokyo",
            narrate_hint="A second, older profile of the same person. Shorter and less specific than the "
            "first. Never call it a duplicate.",
        ),
        # A big company and a Director title, with only PyTorch to be found by.
        _candidate(
            "t3wg8nlp", "Harper", "Sheffield",
            headline="Director, AI Strategy · data platforms",
            seniority="Director",
            city="Tokyo", country="JP", job_function="ML",
            industry="Internet",
            updated="2026-06-25",
            connections=2100,
            positions=[
                _position("Director, AI Strategy", "Halcyon Technologies", size="10,001+",
                          start=(2022, 6), location=TOKYO),
                _position("Senior Program Manager", "Oakhelm Dynamics", size="5,001-10,000",
                          start=(2018, 9), end=(2022, 5), location=TOKYO),
            ],
            skills=[Skill("PyTorch", 9), Skill("SQL", 26), Skill("Snowflake", 14)],
            contacts=[_INMAIL_EN],
            trap="shiny-but-unqualified",
            jd="ml-platform-tokyo",
            narrate_hint="AI strategy and running an organization only — no hands-on platform work, no "
            "Kubernetes operations.",
        ),
        # Nothing since 2024-03. Believing the open-ended current role gives 48 months of
        # ML platform work; counting to the last update gives 19.
        _candidate(
            "f5pq2mhb", "Nova", "Ingram",
            headline="ML Platform Engineer · Kubernetes, PyTorch",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            updated="2024-03-15",
            connections=502,
            positions=[
                _position("ML Platform Engineer", "Glasswing Platform", size="201-500",
                          start=(2022, 8), location=TOKYO),
                _position("Data Platform Engineer", "Draftwell Systems", size="51-200",
                          start=(2019, 6), end=(2022, 7), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 21), Skill("Python", 25), Skill("PyTorch", 12),
                    Skill("MLOps", 9)],
            trap="stale-profile",
            jd="ml-platform-tokyo",
            narrate_hint="Do not say the profile is out of date, and do not assert the role still holds — "
            "last_updated_at says when it was written.",
        ),
        # An 18-month gap. Same total tenure as the control (89 months).
        _candidate(
            "c7rv4bkt", "Ellis", "Norwood",
            headline="ML Platform Engineer · PyTorch, Kubernetes",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-05-30",
            connections=764,
            positions=[
                _position("ML Platform Engineer", "Ironvale Works", size="501-1,000",
                          start=(2022, 8), location=TOKYO),
                _position("Machine Learning Engineer", "Peatmoor Platform", size="201-500",
                          start=(2017, 9), end=(2021, 2), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 19), Skill("Python", 23), Skill("PyTorch", 15),
                    Skill("MLOps", 8)],
            contacts=[_INMAIL_EN],
            trap="employment-gap",
            jd="ml-platform-tokyo",
            narrate_hint="Give no reason for the gap between 2021-02 and 2022-08 — facts only, no invented "
            "reason.",
        ),
        # CTO of three people. By title alone, level with the control at 5,000.
        _candidate(
            "n2bs6zqf", "Blake", "Dunmore",
            headline="CTO · ML infrastructure, PyTorch",
            seniority="CXO",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-03-04",
            connections=431,
            positions=[
                _position("Chief Technology Officer", "Cindershift", size="1-10",
                          start=(2023, 4), location=TOKYO),
                _position("Backend Engineer", "Draftwell Works", size="51-200",
                          start=(2020, 1), end=(2023, 3), location=TOKYO),
            ],
            skills=[Skill("PyTorch", 11), Skill("Python", 14), Skill("Kubernetes", 7)],
            contacts=[_INMAIL_EN],
            trap="inflated-title",
            jd="ml-platform-tokyo",
            narrate_hint="CTO of a three-person team, said with the company size. Write what was actually done "
            "and do not inflate the title.",
        ),
        # ─── backend-seoul-ko ────────────────────────────────────────────
        # One name, two people — the opposite of duplicate-profile. Both can be candidates.
        _candidate(
            "k8vq3mrt", "서연", "강",
            headline="서버 개발자 · Rust, Kafka",
            seniority="Senior",
            language="ko",
            updated="2026-06-14",
            connections=617,
            positions=[
                _position("서버 개발자", "Mossbank Networks", size="201-500",
                          start=(2019, 3)),
            ],
            skills=[Skill("Rust", 22), Skill("Kafka", 15), Skill("PostgreSQL", 12)],
            contacts=[_INMAIL_KO],
            trap="same-name",
            pair_with="urn:li:person:z4hn7cwp",
            jd="backend-seoul-ko",
            narrate_hint="Be specific about company, dates, and technology so this is distinguishable from the "
            "other profile of the same name. Never mention the shared name. Write in Korean.",
        ),
        _candidate(
            "z4hn7cwp", "서연", "강",
            headline="백엔드 개발자 · Rust, Go",
            seniority="Senior",
            language="ko",
            industry="Internet",
            updated="2026-01-27",
            connections=302,
            positions=[
                _position("백엔드 개발자", "Foldgate Works", size="51-200",
                          start=(2021, 7)),
            ],
            skills=[Skill("Rust", 16), Skill("Go", 13), Skill("PostgreSQL", 9)],
            trap="same-name",
            pair_with="urn:li:person:k8vq3mrt",
            jd="backend-seoul-ko",
            narrate_hint="Be specific about company, dates, and technology so this is distinguishable from the "
            "other profile of the same name. Never mention the shared name. Write in Korean.",
        ),
        # Skills identical to the control, city different. That is the whole trap.
        _candidate(
            "s6dt2jkv", "준서", "함",
            headline="서버 개발자 · Rust, 분산 시스템",
            seniority="Senior",
            city="Berlin", country="DE",
            language="ko",
            updated="2026-05-08",
            connections=848,
            positions=[
                _position("서버 개발자", "Nordwind Systems", size="501-1,000",
                          start=(2019, 5), location=BERLIN),
                _position("백엔드 개발자", "Emberpath Labs", size="51-200",
                          start=(2016, 2), end=(2019, 4), location=BERLIN),
            ],
            skills=[Skill("Rust", 29), Skill("Distributed Systems", 19),
                    Skill("PostgreSQL", 14)],
            contacts=[_INMAIL_KO],
            trap="location-mismatch",
            jd="backend-seoul-ko",
            narrate_hint="Do not hide that they live in Berlin, and no speculation about relocation. Write in "
            "Korean.",
        ),
        _candidate(
            "w3gm8rbq", "지훈", "류",
            headline="서버 개발자 · Rust, Kafka, MSA",
            seniority="Senior",
            language="ko",
            updated="2026-06-19",
            connections=559,
            positions=[
                _position("서버 개발자", "Larkfield Networks", size="201-500",
                          start=(2019, 8)),
                _position("백엔드 개발자", "Junipex Labs", size="11-50",
                          start=(2016, 11), end=(2019, 7)),
            ],
            skills=[Skill("Rust", 26), Skill("Kafka", 17), Skill("PostgreSQL", 13)],
            contacts=[_INMAIL_KO],
            trap="korean-only-profile",
            jd="backend-seoul-ko",
            narrate_hint="Write the whole profile in Korean. No English sentences.",
        ),
        # A strong fit with no way to reach them. The trap needs them in the top k, so only
        # contacts differs from the control.
        _candidate(
            "y5cf9pxn", "민준", "심",
            headline="서버 개발자 · Rust, 분산 시스템",
            seniority="Senior",
            language="ko",
            updated="2026-07-27",
            connections=1088,
            positions=[
                _position("서버 개발자", "Oakhelm Systems", size="1,001-5,000",
                          start=(2020, 2)),
                _position("백엔드 개발자", "Halcyon Labs", size="51-200",
                          start=(2016, 9), end=(2020, 1)),
            ],
            skills=[Skill("Rust", 31), Skill("Distributed Systems", 21),
                    Skill("PostgreSQL", 15)],
            trap="no-contact",
            jd="backend-seoul-ko",
            narrate_hint="Do not say they cannot be reached — an empty contacts list carries that. Write in "
            "Korean.",
        ),
    ] + _rank_inversion_pair()


def _controls() -> list[Candidate]:
    """The 11 controls. Three trap kinds are their own control and have none here.

    Each differs from its trap on **exactly one axis**. Differ on a second and there is no
    telling what the judgment turned on — that is not a control, just another person.
    """
    return [
        # The headline leads with Rust, like the trap. What differs is a position where
        # Rust was actually used, which makes "Rust in the headline means bait" wrong.
        #
        # **This axis lives in `positions.description`**, which the prose layer fills, so a
        # test written here would pass while asserting nothing. Only `narrate_hint` carries
        # the axis until then.
        _candidate(
            "g9wr5tvb", "Devon", "Grantham",
            headline="Backend Engineer · Rust, Python",
            seniority="Senior",
            updated="2026-04-02",
            connections=736,
            positions=[
                _position("Backend Engineer", "Glasswing Systems", size="201-500",
                          start=(2024, 1)),
                _position("Software Engineer", "Nightjar Works", size="51-200",
                          start=(2020, 6), end=(2023, 12)),
            ],
            skills=[Skill("Rust", 17), Skill("Python", 28)],
            contacts=[_INMAIL_EN],
            control_for="headline-bait",
            jd="backend-rust",
            narrate_hint="Be specific about real Rust work in the most recent position, and let it show that "
            "the Rust history is under three years.",
        ),
        # Rust in the skills backed by real work, which makes "Rust in the skills means
        # suspicion" wrong.
        #
        # **Trap and control are indistinguishable in the structure layer** — both hold Rust
        # in the skills and both have empty descriptions. They separate in the prose.
        _candidate(
            "p4jn8qhd", "Finley", "Hollis",
            headline="Senior Backend Engineer · Rust, distributed systems",
            seniority="Senior",
            updated="2026-07-15",
            connections=1342,
            positions=[
                _position("Senior Backend Engineer", "Mossbank Dynamics", size="501-1,000",
                          start=(2021, 3)),
                _position("Backend Engineer", "Arborline Labs", size="51-200",
                          start=(2017, 5), end=(2021, 2)),
            ],
            skills=[Skill("Rust", 29), Skill("Distributed Systems", 17),
                    Skill("PostgreSQL", 11)],
            contacts=[_INMAIL_EN],
            control_for="skills-without-evidence",
            jd="backend-rust",
            narrate_hint="Write both positions so the Rust in the skills is verifiable from them.",
        ),
        # Two positions like the trap, not overlapping — which makes "two positions means
        # discount them" wrong. 94 months.
        _candidate(
            "x7bk3mfz", "Quinn", "Westbrook",
            headline="Senior Backend Engineer · Rust, Go",
            seniority="Senior",
            updated="2026-03-21",
            connections=905,
            positions=[
                _position("Backend Engineer", "Kelpstone Labs", size="201-500",
                          start=(2016, 1), end=(2019, 12)),
                _position("Senior Backend Engineer", "Ironvale Platform", size="501-1,000",
                          start=(2020, 1), end=(2023, 12)),
            ],
            skills=[Skill("Rust", 25), Skill("Go", 15), Skill("Distributed Systems", 13)],
            # The axis is the dates, not the contact. Leaving contacts empty would make
            # this person satisfy `no-contact` as well, turning a clean comparison into a
            # second trap.
            contacts=[_INMAIL_EN],
            control_for="overlapping-tenure",
            jd="backend-rust",
            narrate_hint="The dates say the positions do not overlap. Do not stress in prose that they were "
            "not concurrent.",
        ),
        # The same strength and the same span structure. Only open_to_work differs.
        _candidate(
            "r2vt6nsw", "Sage", "Radcliffe",
            headline="Staff Backend Engineer · Rust, distributed systems",
            seniority="Senior",
            updated="2026-08-01",
            connections=1493,
            positions=[
                _position("Staff Backend Engineer", "Ironvale Dynamics", size="1,001-5,000",
                          start=(2022, 2)),
                _position("Backend Engineer", "Foldgate Systems", size="201-500",
                          start=(2018, 1), end=(2022, 1)),
            ],
            skills=[Skill("Rust", 32), Skill("Distributed Systems", 23),
                    Skill("Kubernetes", 11)],
            contacts=[_INMAIL_EN],
            control_for="strong-but-not-open",
            jd="backend-rust",
            narrate_hint="Do not write about looking for work — the open_to_work flag says it.",
        ),
        # A big company and a Director title, like the trap, with the must-have actually
        # done — which makes "a senior title means no practice" wrong.
        _candidate(
            "j8mq4dhr", "Iris", "Pemberton",
            headline="Director, ML Platform · Kubernetes, PyTorch",
            seniority="Director",
            city="Tokyo", country="JP", job_function="ML",
            industry="Internet",
            updated="2026-06-08",
            connections=1876,
            positions=[
                _position("Director, ML Platform", "Oakhelm Technologies", size="10,001+",
                          start=(2021, 11), location=TOKYO),
                _position("Engineering Manager, ML Infrastructure", "Quantile Systems",
                          size="1,001-5,000", start=(2017, 3), end=(2021, 10),
                          location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 31), Skill("MLOps", 22), Skill("Python", 27),
                    Skill("PyTorch", 16)],
            contacts=[_INMAIL_EN],
            control_for="shiny-but-unqualified",
            jd="ml-platform-tokyo",
            narrate_hint="Be specific about building the ML platform personally — not only about running an "
            "organization.",
        ),
        # Same positions and dates as the trap, updated recently. Nothing is an estimate.
        _candidate(
            "q3fw7bkp", "Lane", "Osgood",
            headline="ML Platform Engineer · Kubernetes, PyTorch",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-07-20",
            connections=641,
            positions=[
                _position("ML Platform Engineer", "Emberpath Platform", size="201-500",
                          start=(2022, 8), location=TOKYO),
                _position("Data Platform Engineer", "Junipex Works", size="51-200",
                          start=(2019, 6), end=(2022, 7), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 20), Skill("Python", 24), Skill("PyTorch", 13),
                    Skill("MLOps", 10)],
            contacts=[_INMAIL_EN],
            control_for="stale-profile",
            jd="ml-platform-tokyo",
            narrate_hint="Do not write that the profile was updated recently.",
        ),
        # 89 months like the trap, with the two positions adjacent — the gap has to be
        # measured, not assumed from the shape.
        #
        # The end and the next start are deliberately one month apart: half-open, that is a
        # gap of 1, and the axis this control holds is the gap, so it must not be larger.
        _candidate(
            "v6hd2ptq", "Piper", "Jessup",
            headline="ML Platform Engineer · PyTorch, Kubernetes",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-05-16",
            connections=812,
            positions=[
                _position("ML Platform Engineer", "Nightjar Dynamics", size="501-1,000",
                          start=(2022, 8), location=TOKYO),
                _position("Machine Learning Engineer", "Cindershift Networks",
                          size="201-500", start=(2019, 3), end=(2022, 8), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 18), Skill("Python", 22), Skill("PyTorch", 14),
                    Skill("MLOps", 9)],
            contacts=[_INMAIL_EN],
            control_for="employment-gap",
            jd="ml-platform-tokyo",
            narrate_hint="The dates say the positions are continuous. Do not stress the absence of a gap.",
        ),
        # CXO like the trap, at a different company size. By title alone, identical.
        _candidate(
            "l5nz8crt", "Orion", "Brennan",
            headline="CTO · ML platform, Kubernetes",
            seniority="CXO",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-02-26",
            connections=2643,
            positions=[
                _position("Chief Technology Officer", "Nordwind Technologies",
                          size="5,001-10,000", start=(2021, 9), location=TOKYO),
                _position("VP of Engineering", "Kelpstone Networks", size="1,001-5,000",
                          start=(2016, 4), end=(2021, 8), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 26), Skill("MLOps", 19), Skill("Python", 21),
                    Skill("PyTorch", 12)],
            contacts=[_INMAIL_EN],
            control_for="inflated-title",
            jd="ml-platform-tokyo",
            narrate_hint="CTO of a five-thousand-person company, said with the size.",
        ),
        # Same skills and dates as the trap, in Seoul.
        _candidate(
            "b3tq9wmk", "하은", "성",
            headline="서버 개발자 · Rust, 분산 시스템",
            seniority="Senior",
            language="ko",
            updated="2026-05-08",
            connections=851,
            positions=[
                _position("서버 개발자", "Nordwind Works", size="501-1,000",
                          start=(2019, 5)),
                _position("백엔드 개발자", "Emberpath Systems", size="51-200",
                          start=(2016, 2), end=(2019, 4)),
            ],
            skills=[Skill("Rust", 28), Skill("Distributed Systems", 18),
                    Skill("PostgreSQL", 13)],
            contacts=[_INMAIL_KO],
            control_for="location-mismatch",
            jd="backend-seoul-ko",
            narrate_hint="Let it show they work in Seoul. Do not raise location as an issue. Write in Korean.",
        ),
        # A Korean name in Seoul like the trap, with an English profile — the mail's
        # language follows `profile_language`, not the name.
        _candidate(
            "n7kv4jsb", "도윤", "반",
            headline="Senior Backend Engineer · Rust, Kafka",
            seniority="Senior",
            updated="2026-06-19",
            connections=573,
            positions=[
                _position("Senior Backend Engineer", "Larkfield Dynamics", size="201-500",
                          start=(2019, 8)),
                _position("Backend Engineer", "Junipex Systems", size="11-50",
                          start=(2016, 11), end=(2019, 7)),
            ],
            skills=[Skill("Rust", 27), Skill("Kafka", 16), Skill("PostgreSQL", 12)],
            contacts=[_INMAIL_EN],
            control_for="korean-only-profile",
            jd="backend-seoul-ko",
            narrate_hint="Write the whole profile in English. No Korean sentences.",
        ),
        # Same skills and tenure as the trap, reachable.
        _candidate(
            "h2wp6tdz", "예준", "위",
            headline="서버 개발자 · Rust, 분산 시스템",
            seniority="Senior",
            language="ko",
            updated="2026-07-27",
            connections=1102,
            positions=[
                _position("서버 개발자", "Oakhelm Labs", size="1,001-5,000",
                          start=(2020, 2)),
                _position("백엔드 개발자", "Halcyon Networks", size="51-200",
                          start=(2016, 9), end=(2020, 1)),
            ],
            skills=[Skill("Rust", 30), Skill("Distributed Systems", 20),
                    Skill("PostgreSQL", 14)],
            contacts=[_REFERRAL_KO],
            control_for="no-contact",
            jd="backend-seoul-ko",
            narrate_hint="Do not write that they can be reached — contacts says it. Write in Korean.",
        ),
    ]


def _clear_fits() -> list[Candidate]:
    """Five clear fits: comfortably past every must-have, with nothing to weigh.

    None for `ml-platform-tokyo`, deliberately: that posting is meant to qualify fewer
    than k and adding a clear fit here blurs what it tests.
    """
    return [
        _candidate(
            "a4rn7qvt", "Alex", "Vance",
            headline="Senior Backend Engineer · Rust, distributed systems",
            seniority="Senior",
            updated="2026-07-08",
            connections=1207,
            positions=[
                _position("Senior Backend Engineer", "Ironvale Systems", size="1,001-5,000",
                          start=(2021, 6)),
                _position("Backend Engineer", "Peatmoor Labs", size="201-500",
                          start=(2017, 8), end=(2021, 5)),
            ],
            skills=[Skill("Rust", 36), Skill("Distributed Systems", 25),
                    Skill("PostgreSQL", 15), Skill("Kubernetes", 12)],
            contacts=[_INMAIL_EN],
            verdict="clear-fit",
            jd="backend-rust",
            narrate_hint="Be specific about the systems built in Rust, with their scale.",
        ),
        _candidate(
            "e8mk2wpb", "Reese", "Whitlock",
            headline="Backend Engineer · Rust, Kafka",
            seniority="Senior",
            city="Seongnam",
            industry="Financial Services",
            updated="2026-06-30",
            connections=984,
            positions=[
                _position("Backend Engineer", "Finlogic Systems", size="501-1,000",
                          start=(2020, 9)),
                _position("Software Engineer", "Draftwell Networks", size="51-200",
                          start=(2018, 2), end=(2020, 8)),
            ],
            skills=[Skill("Rust", 31), Skill("Kafka", 19),
                    Skill("Distributed Systems", 13)],
            contacts=[_INMAIL_EN],
            verdict="clear-fit",
            jd="backend-rust",
            narrate_hint="Be specific about handling payment traffic in Rust.",
        ),
        _candidate(
            "u5cz9jhr", "Tatum", "Ellery",
            headline="Staff Backend Engineer · Rust platform",
            seniority="Senior",
            updated="2026-08-02",
            connections=1655,
            positions=[
                _position("Staff Backend Engineer", "Mossbank Platform", size="1,001-5,000",
                          start=(2022, 3)),
                _position("Senior Backend Engineer", "Cindershift Systems", size="201-500",
                          start=(2017, 11), end=(2022, 2)),
            ],
            skills=[Skill("Rust", 34), Skill("Distributed Systems", 27),
                    Skill("Kubernetes", 15)],
            contacts=[_REFERRAL_EN],
            verdict="clear-fit",
            jd="backend-rust",
            narrate_hint="Be specific about the Rust work done on the platform team.",
        ),
        _candidate(
            "k6qb3nwd", "채원", "노",
            headline="서버 개발자 · Rust, 분산 시스템",
            seniority="Senior",
            language="ko",
            updated="2026-07-22",
            connections=877,
            positions=[
                _position("서버 개발자", "Halcyon Systems", size="1,001-5,000",
                          start=(2020, 5)),
                _position("백엔드 개발자", "Mossbank Labs", size="51-200",
                          start=(2016, 7), end=(2020, 4)),
            ],
            skills=[Skill("Rust", 33), Skill("Distributed Systems", 21),
                    Skill("Kafka", 16)],
            contacts=[_INMAIL_KO],
            verdict="clear-fit",
            jd="backend-seoul-ko",
            narrate_hint="Write the systems built in Rust with their scale. Write in Korean.",
        ),
        _candidate(
            "w9td4bmq", "시우", "도",
            headline="백엔드 개발자 · Rust, Go",
            seniority="Senior",
            language="ko",
            industry="Internet",
            updated="2026-06-11",
            connections=1043,
            positions=[
                _position("백엔드 개발자", "Foldgate Dynamics", size="501-1,000",
                          start=(2019, 12)),
                _position("서버 개발자", "Nightjar Systems", size="11-50",
                          start=(2017, 3), end=(2019, 11)),
            ],
            skills=[Skill("Rust", 27), Skill("Go", 18), Skill("PostgreSQL", 15),
                    Skill("Distributed Systems", 11)],
            contacts=[_INMAIL_KO],
            verdict="clear-fit",
            jd="backend-seoul-ko",
            narrate_hint="Be specific about the Rust work and the Go work separately. Write in Korean.",
        ),
    ]


def _borderlines() -> list[Candidate]:
    """Twelve borderline people, short on one axis and meeting the rest.

    Borderline does not mean weaker — it means **the shortfall has to be named**. Each is
    short on exactly one axis, and `narrate_hint` makes that axis visible in the prose;
    erase it and a borderline person reads as a clear fit.
    """
    return [
        # ─── backend-rust ────────────────────────────────────────────────
        # 86 months total, only the last 43 in Rust. The question is what the years were spent on.
        _candidate(
            "d3hv8ktp", "Jade", "Underhill",
            headline="Backend Engineer · Rust, Python",
            seniority="Senior",
            updated="2026-05-19",
            connections=712,
            positions=[
                _position("Backend Engineer", "Emberpath Networks", size="201-500",
                          start=(2023, 1)),
                _position("Software Engineer", "Glasswing Labs", size="51-200",
                          start=(2019, 5), end=(2022, 12)),
            ],
            skills=[Skill("Rust", 18), Skill("Python", 24)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="backend-rust",
            narrate_hint="Let it show the Rust is three and a half years, all in the most recent position.",
        ),
        # Seven years of Rust, in systems software rather than servers.
        _candidate(
            "f7pw2mrj", "Piper", "Ellery",
            headline="Systems Engineer · Rust, Linux",
            seniority="Senior",
            job_function="Infrastructure",
            industry="Information Technology and Services",
            updated="2026-04-27",
            connections=638,
            positions=[
                _position("Systems Engineer", "Kelpstone Technologies", size="501-1,000",
                          start=(2019, 2)),
            ],
            skills=[Skill("Rust", 30), Skill("Linux", 26), Skill("Observability", 14)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="backend-rust",
            narrate_hint="Let it show the work was systems software rather than servers or distributed "
            "systems.",
        ),
        # All Rust, 46 months — just under the 48-month must-have. Rounds up, counts short.
        _candidate(
            "m2sq7nvc", "Quinn", "Fairbanks",
            headline="Backend Engineer · Rust, PostgreSQL",
            seniority="Senior",
            updated="2026-07-05",
            connections=559,
            positions=[
                _position("Backend Engineer", "Nordwind Labs", size="51-200",
                          start=(2022, 10)),
            ],
            skills=[Skill("Rust", 23), Skill("Distributed Systems", 12),
                    Skill("PostgreSQL", 9)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="backend-rust",
            narrate_hint="Let it show the whole career is three years and ten months at one company.",
        ),
        # Enough Rust, but three years without writing code.
        _candidate(
            "t8jc3wqz", "Devon", "Whitlock",
            headline="Engineering Manager · Rust backend platform",
            seniority="Manager",
            updated="2026-06-03",
            connections=1387,
            positions=[
                _position("Engineering Manager", "Arborline Technologies",
                          size="1,001-5,000", start=(2023, 5)),
                _position("Senior Backend Engineer", "Peatmoor Dynamics", size="201-500",
                          start=(2018, 10), end=(2023, 4)),
            ],
            skills=[Skill("Rust", 28), Skill("Distributed Systems", 19),
                    Skill("Kubernetes", 13)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="backend-rust",
            narrate_hint="Let it show the last three years were mostly running a team.",
        ),
        # ─── ml-platform-tokyo ───────────────────────────────────────────
        # Tokyo, Kubernetes, Python — but data pipelines rather than an ML platform.
        _candidate(
            "y4bn9khs", "Sage", "Vance",
            headline="Data Platform Engineer · Airflow, Kubernetes",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="Data",
            industry="Internet",
            updated="2026-05-25",
            connections=793,
            positions=[
                _position("Data Platform Engineer", "Quantile Networks", size="501-1,000",
                          start=(2019, 7), location=TOKYO),
            ],
            skills=[Skill("Airflow", 27), Skill("Python", 30), Skill("SQL", 22),
                    Skill("Kubernetes", 11), Skill("PyTorch", 7)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="ml-platform-tokyo",
            narrate_hint="Let it show they built data pipelines rather than an ML training or serving "
            "platform.",
        ),
        # The ML platform history is right; the city is Seoul.
        _candidate(
            "g2vt6mqd", "Iris", "Calloway",
            headline="ML Platform Engineer · Kubernetes, PyTorch",
            seniority="Senior",
            job_function="ML",
            updated="2026-07-13",
            connections=1024,
            positions=[
                _position("ML Platform Engineer", "Oakhelm Platform", size="1,001-5,000",
                          start=(2020, 11)),
            ],
            skills=[Skill("Kubernetes", 29), Skill("MLOps", 21), Skill("Python", 26),
                    Skill("PyTorch", 17)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="ml-platform-tokyo",
            narrate_hint="Do not hide that they work in Seoul, and do not speculate about willingness to move.",
        ),
        # Eight years of ML infrastructure, the last four managing rather than operating.
        _candidate(
            "s9wk4bpr", "Blake", "Merrick",
            headline="Engineering Manager, ML Infrastructure · PyTorch",
            seniority="Manager",
            city="Tokyo", country="JP", job_function="ML",
            industry="Internet",
            updated="2026-06-17",
            connections=1571,
            positions=[
                _position("Engineering Manager, ML Infrastructure", "Nordwind Dynamics",
                          size="5,001-10,000", start=(2022, 1), location=TOKYO),
                _position("Machine Learning Engineer", "Junipex Technologies",
                          size="201-500", start=(2018, 5), end=(2021, 12), location=TOKYO),
            ],
            skills=[Skill("Python", 29), Skill("PyTorch", 20), Skill("MLOps", 17),
                    Skill("Kubernetes", 8)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="ml-platform-tokyo",
            narrate_hint="Let it show the last four years were mostly running a team, with no hands-on "
            "Kubernetes.",
        ),
        # 43 months of ML platform work — on the tenure boundary.
        _candidate(
            "c5nq8jtv", "Lane", "Ashby",
            headline="ML Platform Engineer · Kubernetes, PyTorch",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-07-31",
            connections=685,
            positions=[
                _position("ML Platform Engineer", "Cindershift Labs", size="51-200",
                          start=(2023, 1), location=TOKYO),
                _position("Backend Engineer", "Halcyon Platform", size="201-500",
                          start=(2019, 9), end=(2022, 12), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 22), Skill("Python", 25), Skill("PyTorch", 11),
                    Skill("MLOps", 7)],
            contacts=[_INMAIL_EN],
            verdict="borderline",
            jd="ml-platform-tokyo",
            narrate_hint="Let it show the ML platform history is three and a half years.",
        ),
        # ─── backend-seoul-ko ────────────────────────────────────────────
        # Seven years of backend, the last 39 months in Rust, Java before that.
        _candidate(
            "z6hm2wct", "은서", "고",
            headline="서버 개발자 · Rust, Java",
            seniority="Senior",
            language="ko",
            updated="2026-05-04",
            connections=496,
            positions=[
                _position("서버 개발자", "Draftwell Platform", size="201-500",
                          start=(2023, 5)),
                _position("백엔드 개발자", "Mossbank Works", size="11-50",
                          start=(2019, 4), end=(2023, 4)),
            ],
            skills=[Skill("Rust", 15), Skill("Java", 26), Skill("PostgreSQL", 12)],
            contacts=[_INMAIL_KO],
            verdict="borderline",
            jd="backend-seoul-ko",
            narrate_hint="Let it show the Rust is a little over three years, all in the most recent position. "
            "Write in Korean.",
        ),
        # Six years of Rust, all at 10–50 people. No experience at scale.
        _candidate(
            "p3kd7nwq", "하윤", "표",
            headline="서버 개발자 · Rust, Go",
            seniority="Senior",
            city="Seongnam",
            language="ko",
            updated="2026-06-21",
            connections=604,
            positions=[
                _position("서버 개발자", "Arborline", size="11-50", start=(2020, 6)),
                _position("백엔드 개발자", "Nightjar Platform", size="11-50",
                          start=(2018, 1), end=(2020, 5)),
            ],
            skills=[Skill("Rust", 21), Skill("PostgreSQL", 14), Skill("Go", 10)],
            contacts=[_INMAIL_KO],
            verdict="borderline",
            jd="backend-seoul-ko",
            narrate_hint="Say plainly that the companies were small. Do not inflate the traffic. Write in "
            "Korean.",
        ),
        # Six years of infrastructure and two of service backend. Rust on internal tools.
        _candidate(
            "v7tb4mkj", "소율", "명",
            headline="인프라 엔지니어 · Kubernetes, Rust",
            seniority="Senior",
            language="ko",
            job_function="Infrastructure",
            industry="Information Technology and Services",
            updated="2026-04-15",
            connections=741,
            positions=[
                _position("인프라 엔지니어", "Ironvale Labs", size="1,001-5,000",
                          start=(2020, 3)),
                _position("백엔드 개발자", "Emberpath Dynamics", size="51-200",
                          start=(2018, 2), end=(2020, 2)),
            ],
            skills=[Skill("Rust", 16), Skill("Kubernetes", 27), Skill("Terraform", 20),
                    Skill("Linux", 22)],
            contacts=[_INMAIL_KO],
            verdict="borderline",
            jd="backend-seoul-ko",
            narrate_hint="Let it show the work was mostly infrastructure, with two years of service backend. "
            "Write in Korean.",
        ),
        # Eight years of Rust backend, the last two leading a team.
        _candidate(
            "q8jw3rvh", "예은", "구",
            headline="개발팀장 · Rust, 분산 시스템",
            seniority="Manager",
            language="ko",
            industry="Internet",
            updated="2026-07-19",
            connections=1298,
            positions=[
                _position("개발팀장", "Foldgate Technologies", size="1,001-5,000",
                          start=(2024, 5)),
                _position("서버 개발자", "Kelpstone Systems", size="501-1,000",
                          start=(2016, 3), end=(2024, 4)),
            ],
            skills=[Skill("Rust", 24), Skill("PostgreSQL", 18),
                    Skill("Distributed Systems", 15)],
            contacts=[_INMAIL_KO],
            verdict="borderline",
            jd="backend-seoul-ko",
            narrate_hint="Let it show the last two years were mostly leading a team. Write in Korean.",
        ),
    ]


def _clear_misses() -> list[Candidate]:
    """Twenty clear misses, five per posting.

    Not weaker, **different**. An eight-year frontend engineer misses a backend
    Rust posting because of what they have done, not how well. So
    Each `narrate_hint` asks for good writing about their own field and forbids only the
    posting's must-have. One line of "I have done a little Rust too" and the miss stops
    being clear.

    The five against `blockchain-solidity` are adjacent by headline alone, with Solidity
    nowhere. Nobody qualifying is that posting's purpose, and saying so is only a test when
    something plausible is in front of you.
    """
    return [
        # ─── backend-rust ────────────────────────────────────────────────
        _candidate(
            "r5mv2qkb", "Jordan", "Underhill",
            headline="Senior Frontend Engineer · React, TypeScript",
            seniority="Senior",
            job_function="Frontend",
            industry="Internet",
            updated="2026-06-28",
            connections=1132,
            positions=[
                _position("Senior Frontend Engineer", "Glasswing Networks",
                          size="501-1,000", start=(2018, 6)),
            ],
            skills=[Skill("TypeScript", 38), Skill("React", 33),
                    Skill("Accessibility", 17)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="backend-rust",
            narrate_hint="Frontend work only — no Rust, no backend work.",
        ),
        _candidate(
            "w8kq4jtd", "Casey", "Ellery",
            headline="iOS Engineer · Swift",
            seniority="Senior",
            job_function="Mobile",
            industry="Internet",
            updated="2026-03-17",
            connections=657,
            positions=[
                _position("iOS Engineer", "Nightjar Platform", size="11-50",
                          start=(2019, 1)),
            ],
            skills=[Skill("Swift", 34), Skill("iOS", 28), Skill("Kotlin", 12)],
            verdict="clear-miss",
            jd="backend-rust",
            narrate_hint="iOS work only — no Rust, no server backend.",
        ),
        _candidate(
            "h3nb7wcp", "Riley", "Vance",
            headline="Data Engineering Manager · Spark, Airflow",
            seniority="Manager",
            city="Seongnam", job_function="Data",
            industry="Financial Services",
            updated="2026-05-06",
            connections=1408,
            positions=[
                _position("Data Engineering Manager", "Finlogic Dynamics",
                          size="1,001-5,000", start=(2021, 2)),
                _position("Data Engineer", "Junipex Networks", size="201-500",
                          start=(2016, 5), end=(2021, 1)),
            ],
            skills=[Skill("Spark", 31), Skill("Python", 27), Skill("SQL", 24)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="backend-rust",
            narrate_hint="Data pipeline work only — no Rust.",
        ),
        _candidate(
            "s6tw9mqk", "Morgan", "Osgood",
            headline="Backend Engineer · Java, PostgreSQL",
            seniority="Entry",
            updated="2026-07-24",
            connections=214,
            positions=[
                _position("Backend Engineer", "Larkfield Systems", size="51-200",
                          start=(2024, 8)),
            ],
            skills=[Skill("Java", 13), Skill("PostgreSQL", 9)],
            verdict="clear-miss",
            jd="backend-rust",
            narrate_hint="Let it show two years of experience. No Rust.",
        ),
        _candidate(
            "c9qj3vhn", "Avery", "Whitlock",
            headline="Analytics Engineer · dbt, Snowflake",
            seniority="Senior",
            city="San Francisco", country="US", job_function="Data",
            industry="Information Technology and Services",
            updated="2026-04-09",
            connections=889,
            positions=[
                _position("Analytics Engineer", "Oakhelm Works", size="501-1,000",
                          start=(2019, 3), location=SF),
            ],
            skills=[Skill("dbt", 26), Skill("SQL", 32), Skill("Snowflake", 19)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="backend-rust",
            narrate_hint="Analytics engineering only — no Rust, no backend.",
        ),
        # ─── ml-platform-tokyo ───────────────────────────────────────────
        _candidate(
            "n4vd8ktr", "Kai", "Brennan",
            headline="Frontend Engineer · Vue, TypeScript",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="Frontend",
            industry="Internet",
            updated="2026-06-05",
            connections=742,
            positions=[
                _position("Frontend Engineer", "Quantile Works", size="201-500",
                          start=(2019, 9), location=TOKYO),
            ],
            skills=[Skill("Vue", 31), Skill("TypeScript", 27), Skill("CSS", 19)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="ml-platform-tokyo",
            narrate_hint="Frontend work only — no ML platform, no Kubernetes operations.",
        ),
        _candidate(
            "b7mk2qwz", "Nova", "Radcliffe",
            headline="ML Engineer · PyTorch",
            seniority="Entry",
            city="Tokyo", country="JP", job_function="ML",
            updated="2026-07-16",
            connections=196,
            positions=[
                _position("ML Engineer", "Emberpath Works", size="51-200",
                          start=(2025, 3), location=TOKYO),
            ],
            skills=[Skill("PyTorch", 9), Skill("Python", 12)],
            verdict="clear-miss",
            jd="ml-platform-tokyo",
            narrate_hint="Let it show a year and a half of experience. No platform operations.",
        ),
        _candidate(
            "t2wq7jbd", "Ellis", "Sheffield",
            headline="Data Analyst · SQL, Snowflake",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="Data",
            industry="Information Technology and Services",
            updated="2026-02-19",
            connections=613,
            positions=[
                _position("Data Analyst", "Nordwind Platform", size="5,001-10,000",
                          start=(2018, 11), location=TOKYO),
            ],
            skills=[Skill("SQL", 33), Skill("Snowflake", 22), Skill("dbt", 15)],
            verdict="clear-miss",
            jd="ml-platform-tokyo",
            narrate_hint="Analytics only — no platform engineering, no Kubernetes.",
        ),
        _candidate(
            "k5rn9tvc", "Harper", "Pemberton",
            headline="Network Engineer · Linux, Networking",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="Infrastructure",
            industry="Information Technology and Services",
            updated="2026-05-21",
            connections=531,
            positions=[
                _position("Network Engineer", "Peatmoor Technologies", size="1,001-5,000",
                          start=(2017, 4), location=TOKYO),
            ],
            skills=[Skill("Networking", 29), Skill("Linux", 26),
                    Skill("Observability", 14)],
            verdict="clear-miss",
            jd="ml-platform-tokyo",
            narrate_hint="Network operations only — no ML, no PyTorch.",
        ),
        _candidate(
            "f8jt3nwq", "Quinn", "Lockhart",
            headline="Android Engineer · Kotlin",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="Mobile",
            industry="Internet",
            updated="2026-06-13",
            connections=804,
            positions=[
                _position("Android Engineer", "Peatmoor Networks", size="501-1,000",
                          start=(2018, 2), location=TOKYO),
            ],
            skills=[Skill("Kotlin", 32), Skill("Android", 29), Skill("Flutter", 11)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="ml-platform-tokyo",
            narrate_hint="Android work only — no ML platform.",
        ),
        # ─── backend-seoul-ko ────────────────────────────────────────────
        _candidate(
            "d6qw4mkt", "다은", "하",
            headline="프론트엔드 개발자 · React",
            seniority="Senior",
            language="ko", job_function="Frontend",
            industry="Internet",
            updated="2026-05-28",
            connections=922,
            positions=[
                _position("프론트엔드 개발자", "Cindershift Networks", size="201-500",
                          start=(2018, 8)),
            ],
            skills=[Skill("React", 33), Skill("TypeScript", 29), Skill("CSS", 18)],
            contacts=[_INMAIL_KO],
            verdict="clear-miss",
            jd="backend-seoul-ko",
            narrate_hint="Frontend work only — no Rust, no backend. Write in Korean.",
        ),
        _candidate(
            "v3bn8jhq", "지우", "남",
            headline="백엔드 개발자 · Python",
            seniority="Entry",
            language="ko",
            updated="2026-08-05",
            connections=178,
            positions=[
                _position("백엔드 개발자", "Kelpstone Platform", size="11-50",
                          start=(2025, 7)),
            ],
            skills=[Skill("Python", 9), Skill("PostgreSQL", 6)],
            verdict="clear-miss",
            jd="backend-seoul-ko",
            narrate_hint="Let it show a little over a year of experience. No Rust. Write in Korean.",
        ),
        _candidate(
            "g7kt2wvn", "현우", "설",
            headline="데이터 엔지니어 · Airflow",
            seniority="Senior",
            language="ko", job_function="Data",
            updated="2026-04-23",
            connections=1067,
            positions=[
                _position("데이터 엔지니어", "Ironvale Technologies", size="1,001-5,000",
                          start=(2017, 10)),
            ],
            skills=[Skill("Airflow", 28), Skill("Python", 25), Skill("SQL", 21)],
            contacts=[_INMAIL_KO],
            verdict="clear-miss",
            jd="backend-seoul-ko",
            narrate_hint="Data pipeline work only — no Rust. Write in Korean.",
        ),
        _candidate(
            "j4mq9btr", "지호", "현",
            headline="모바일 개발자 · Flutter",
            seniority="Senior",
            language="ko", job_function="Mobile",
            industry="Internet",
            updated="2026-07-01",
            connections=486,
            positions=[
                _position("모바일 개발자", "Mossbank Technologies", size="501-1,000",
                          start=(2019, 6)),
            ],
            skills=[Skill("Flutter", 29), Skill("Kotlin", 18), Skill("React Native", 12)],
            verdict="clear-miss",
            jd="backend-seoul-ko",
            narrate_hint="Mobile work only — no Rust, no server backend. Write in Korean.",
        ),
        _candidate(
            "p9wc3ktb", "주원", "강",
            headline="인프라 팀장 · Linux, AWS",
            seniority="Manager",
            language="ko", job_function="Infrastructure",
            industry="Information Technology and Services",
            updated="2026-03-11",
            connections=1345,
            positions=[
                _position("인프라 팀장", "Foldgate Labs", size="1,001-5,000",
                          start=(2016, 2)),
            ],
            skills=[Skill("Linux", 31), Skill("AWS", 26), Skill("Terraform", 19)],
            contacts=[_INMAIL_KO],
            verdict="clear-miss",
            jd="backend-seoul-ko",
            narrate_hint="Infrastructure operations only — no Rust, no service backend. Write in Korean.",
        ),
        # ─── blockchain-solidity ─────────────────────────────────────────
        # All five are adjacent by headline only. Solidity is in no skill and no title.
        _candidate(
            "r7nk4qwj", "Alex", "Merrick",
            headline="Backend Engineer · fintech payments, web3 curious",
            seniority="Senior",
            industry="Financial Services",
            updated="2026-06-09",
            connections=978,
            positions=[
                _position("Backend Engineer", "Finlogic Labs", size="501-1,000",
                          start=(2019, 2)),
            ],
            skills=[Skill("Java", 31), Skill("PostgreSQL", 26), Skill("Kafka", 14)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="blockchain-solidity",
            narrate_hint="Payment backend only — no smart contract work.",
        ),
        _candidate(
            "m3tb8vqd", "Tatum", "Hollis",
            headline="Data Engineer · on-chain analytics",
            seniority="Senior",
            job_function="Data",
            industry="Financial Services",
            updated="2026-05-14",
            connections=655,
            positions=[
                _position("Data Engineer", "Quantile Dynamics", size="201-500",
                          start=(2020, 4)),
            ],
            skills=[Skill("SQL", 29), Skill("Python", 27), Skill("Spark", 16)],
            verdict="clear-miss",
            jd="blockchain-solidity",
            narrate_hint="On-chain data analysis only — no smart contract work.",
        ),
        _candidate(
            "y2qj7wkn", "Reese", "Norwood",
            headline="Backend Engineer · crypto exchange APIs",
            seniority="Entry",
            city="Berlin", country="DE",
            industry="Financial Services",
            updated="2026-07-29",
            connections=241,
            positions=[
                _position("Backend Engineer", "Arborline Networks", size="51-200",
                          start=(2024, 2), location=BERLIN),
            ],
            skills=[Skill("Go", 14), Skill("PostgreSQL", 11)],
            verdict="clear-miss",
            jd="blockchain-solidity",
            narrate_hint="Exchange backend APIs only — no smart contract work.",
        ),
        _candidate(
            "x6vd3ntq", "Jade", "Grantham",
            headline="Site Reliability Engineer · validator node operations",
            seniority="Senior",
            city="Tokyo", country="JP", job_function="Infrastructure",
            industry="Information Technology and Services",
            updated="2026-04-30",
            connections=713,
            positions=[
                _position("Site Reliability Engineer", "Junipex Dynamics", size="201-500",
                          start=(2018, 12), location=TOKYO),
            ],
            skills=[Skill("Kubernetes", 27), Skill("Linux", 24), Skill("Terraform", 21)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="blockchain-solidity",
            narrate_hint="Node operations only — no smart contract work.",
        ),
        _candidate(
            "l4wq9jkb", "Piper", "Voss",
            headline="Frontend Engineer · dApp interfaces",
            seniority="Senior",
            job_function="Frontend",
            industry="Internet",
            updated="2026-06-26",
            connections=846,
            positions=[
                _position("Frontend Engineer", "Draftwell Dynamics", size="51-200",
                          start=(2020, 10)),
            ],
            skills=[Skill("TypeScript", 30), Skill("React", 26), Skill("WebAssembly", 11)],
            contacts=[_INMAIL_EN],
            verdict="clear-miss",
            jd="blockchain-solidity",
            narrate_hint="dApp frontend only — no smart contract work.",
        ),
    ]


def core() -> list[Candidate]:
    """The core 65: 17 traps, 11 controls, 37 for judgment.

    Fresh objects every call. `gen.py` and `truth.py` each call it, and a caller mutating
    the list must not affect the next.
    """
    return _traps() + _controls() + _clear_fits() + _borderlines() + _clear_misses()
