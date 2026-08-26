"""The side tables: education, certifications, languages, and desired arrangement.

**Filled after the roster is complete, not inside the generation loop.** One extra draw
inside `background()` would shift every later person's values. These four are independent
of the existing axes, so each person draws from a stream seeded by their own id — leaving
all 600 names, careers, and skills untouched and keeping one person's values off another.
"""

from common.dates import AS_OF
from common.names import person
from common.rng import seeded
from headhunter import variants
from headhunter.schema import (
    Candidate,
    Skill,
    Certification,
    Education,
    Language,
    OpenToWorkPref,
    Skill,
)

EDUCATION_RATE = 0.80
MASTER_RATE = 0.25

# Invented the same way company roots are. Place and person names collide with real
# schools — `Marlowe College` is one at the University of Kent — and committed data must
# not read as fictional alumni records of a real university.
_SCHOOL_HEAD = [
    "Alderwick", "Briarholt", "Cresthollow", "Dunmere", "Elmstrand", "Fernvault",
    "Gorsecliff", "Hollowbrace", "Ivyquill", "Jarrowfen", "Kelvinmoss", "Lanternwood",
    "Mirefield", "Nettlerow", "Osperdine", "Pinewrack", "Quarrymoss", "Redthistle",
]
_SCHOOL_TAIL = ["University", "Institute of Technology", "College"]

_FIELDS_BY_FUNCTION = {
    "Backend": ["Computer Science", "Software Engineering", "Information Systems"],
    "Frontend": ["Computer Science", "Software Engineering", "Design Computing"],
    "ML": ["Machine Learning", "Statistics", "Mathematics"],
    "Data": ["Statistics", "Information Systems", "Mathematics"],
    "Infrastructure": ["Computer Engineering", "Electrical Engineering", "Computer Science"],
    "Mobile": ["Computer Science", "Software Engineering"],
}
_FIELDS_DEFAULT = ["Computer Science", "Software Engineering"]


def educations_for(candidate: Candidate) -> list[Education]:
    """Zero to two rows. The last degree finishes no later than the first job — not
    because mid-career degrees do not happen, but because one rule can be checked.
    """
    rng = seeded(f"profile:educations:{candidate.id}")
    if rng.random() >= EDUCATION_RATE or not candidate.positions:
        return []

    first_start = min(p.start_year for p in candidate.positions)
    fields = _FIELDS_BY_FUNCTION.get(candidate.job_function, _FIELDS_DEFAULT)
    has_master = rng.random() < MASTER_RATE

    bachelor_end = first_start - (2 if has_master else rng.randint(0, 1))
    school = f"{rng.choice(_SCHOOL_HEAD)} {rng.choice(_SCHOOL_TAIL)}"
    out = [
        Education(
            school_name=school,
            degree_name=rng.choice(["Bachelor of Science", "Bachelor of Engineering"]),
            field_of_study=rng.choice(fields),
            start_year=bachelor_end - 4,
            end_year=bachelor_end,
        )
    ]
    if has_master:
        out.append(
            Education(
                school_name=f"{rng.choice(_SCHOOL_HEAD)} {rng.choice(_SCHOOL_TAIL)}",
                degree_name=rng.choice(["Master of Science", "Master of Engineering"]),
                field_of_study=rng.choice(fields),
                start_year=bachelor_end,
                end_year=bachelor_end + 2,
            )
        )
    return out


CERTIFICATION_RATE = 0.30
LANGUAGE_RATE = 0.70

# Real programs. Unlike companies, certifications are public schemes; inventing one reads false.
_CERTS_BY_FUNCTION = {
    "Backend": [
        ("AWS Certified Solutions Architect - Associate", "Amazon Web Services"),
        ("AWS Certified Developer - Associate", "Amazon Web Services"),
        ("Oracle Certified Professional, Java SE", "Oracle"),
        ("HashiCorp Certified: Terraform Associate", "HashiCorp"),
    ],
    "ML": [
        ("TensorFlow Developer Certificate", "Google"),
        ("Google Cloud Professional Machine Learning Engineer", "Google Cloud"),
        ("AWS Certified Machine Learning - Specialty", "Amazon Web Services"),
    ],
    "Infrastructure": [
        ("Certified Kubernetes Administrator", "Cloud Native Computing Foundation"),
        ("Certified Kubernetes Application Developer", "Cloud Native Computing Foundation"),
        ("HashiCorp Certified: Terraform Associate", "HashiCorp"),
        ("AWS Certified Solutions Architect - Professional", "Amazon Web Services"),
    ],
    "Data": [
        ("Google Cloud Professional Data Engineer", "Google Cloud"),
        ("Microsoft Certified: Azure Data Engineer Associate", "Microsoft"),
    ],
}
_CERTS_BY_FUNCTION["Frontend"] = [
    ("Meta Front-End Developer Professional Certificate", "Meta"),
    ("AWS Certified Cloud Practitioner", "Amazon Web Services"),
]
_CERTS_BY_FUNCTION["Mobile"] = [
    ("Associate Android Developer", "Google"),
    ("AWS Certified Cloud Practitioner", "Amazon Web Services"),
]
_CERTS_DEFAULT = [("AWS Certified Cloud Practitioner", "Amazon Web Services")]

# Frontend and mobile have little certification culture. Letting the data show that beats
# padding a pool.
_CERT_RATE_BY_FUNCTION = {"Frontend": 0.10, "Mobile": 0.12}

# PMP attaches to seniority, not to a function.
_MANAGEMENT = {"Manager", "Director", "CXO", "Owner"}
_MANAGEMENT_CERT = ("Project Management Professional", "Project Management Institute")

_COUNTRY_LANGUAGE = {"KR": "Korean", "JP": "Japanese", "DE": "German", "US": "English"}
_PROFILE_LANGUAGE_NAME = {"en": "English", "ko": "Korean", "ja": "Japanese"}


def certifications_for(candidate: Candidate) -> list[Certification]:
    """Zero to two rows. Frontend has no pool of its own and falls back to the default."""
    rng = seeded(f"profile:certifications:{candidate.id}")
    rate = _CERT_RATE_BY_FUNCTION.get(candidate.job_function, CERTIFICATION_RATE)
    if rng.random() >= rate:
        return []
    pool = list(_CERTS_BY_FUNCTION.get(candidate.job_function, _CERTS_DEFAULT))
    if candidate.seniority in _MANAGEMENT:
        pool.append(_MANAGEMENT_CERT)
    k = 1 if rng.random() < 0.75 else min(2, len(pool))
    return [Certification(name=n, authority=a) for n, a in rng.sample(pool, k)]


def languages_for(candidate: Candidate) -> list[Language]:
    """Zero to three rows. The profile's own language is always present and at working
    level or above — a profile written in English claiming no English contradicts itself.
    """
    rng = seeded(f"profile:languages:{candidate.id}")
    if rng.random() >= LANGUAGE_RATE:
        return []

    primary = _PROFILE_LANGUAGE_NAME[candidate.profile_language]
    out = [
        Language(
            name=primary,
            proficiency=rng.choice(["NATIVE_OR_BILINGUAL", "PROFESSIONAL_WORKING"]),
        )
    ]
    local = _COUNTRY_LANGUAGE.get(candidate.country)
    if local and local != primary and rng.random() < 0.6:
        out.append(Language(name=local, proficiency="NATIVE_OR_BILINGUAL"))
    if primary != "English" and rng.random() < 0.7:
        out.append(
            Language(
                name="English",
                proficiency=rng.choice(["PROFESSIONAL_WORKING", "LIMITED_WORKING"]),
            )
        )
    return out


PREFS_RATE = 0.70

_ADJACENT_TITLE = {
    "Backend": ["Backend Engineer", "Platform Engineer", "Server Engineer"],
    "Frontend": ["Frontend Engineer", "Web Engineer"],
    "ML": ["ML Engineer", "ML Platform Engineer", "Research Engineer"],
    "Data": ["Data Engineer", "Analytics Engineer"],
    "Infrastructure": ["Infrastructure Engineer", "SRE", "Platform Engineer"],
    "Mobile": ["Mobile Engineer", "iOS Engineer", "Android Engineer"],
}


def prefs_for(candidate: Candidate) -> list[OpenToWorkPref]:
    """Zero to two rows, and always zero when `open_to_work` is false — nobody marks
    themselves not looking and then fills in what they are looking for.
    """
    if not candidate.open_to_work:
        return []
    rng = seeded(f"profile:prefs:{candidate.id}")
    if rng.random() >= PREFS_RATE:
        return []

    titles = _ADJACENT_TITLE.get(candidate.job_function, [candidate.job_function + " Engineer"])
    location_type = rng.choices(
        ["On-site", "Hybrid", "Remote"], weights=[25, 50, 25], k=1
    )[0]
    if rng.random() < 0.25:
        year, month = AS_OF
        month += rng.randint(1, 6)
        year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
        start = f"{year:04d}-{month:02d}"
    else:
        start = "immediately"

    k = 1 if rng.random() < 0.6 else min(2, len(titles))
    return [
        OpenToWorkPref(
            desired_title=title,
            location_type=location_type,
            desired_location="Remote" if location_type == "Remote" else candidate.city,
            start_date=start,
            employment_type=rng.choices(["Full-time", "Contract"], weights=[85, 15], k=1)[0],
        )
        for title in rng.sample(titles, k)
    ]


def fill(roster: list[Candidate]) -> list[Candidate]:
    """Fills the four tables across the roster, leaving any existing value alone.

    Hand-written fixture values win over generated ones; a trap needs an exact value.
    """
    for candidate in roster:
        assigned = TRAP_ASSIGNMENTS.get(candidate.id)
        if assigned:
            if "educations" in assigned:
                candidate.educations = assigned["educations"]
            if "certifications" in assigned:
                candidate.certifications = assigned["certifications"]
            if "languages" in assigned:
                candidate.languages = assigned["languages"]
            if "prefs" in assigned:
                candidate.open_to_work_prefs = assigned["prefs"]
            held = getattr(candidate, "_trap", None) or getattr(candidate, "_control_for", None)
            if held:
                raise ValueError(
                    f"{candidate.id} is already held by the fixtures as {held!r} — "
                    "assigning over it would make that trap disappear silently"
                )
            candidate._trap = assigned.get("trap")
            candidate._control_for = assigned.get("control_for")
            candidate._jd = assigned.get("jd")
        if not candidate.educations:
            candidate.educations = educations_for(candidate)
        if not candidate.certifications:
            candidate.certifications = certifications_for(candidate)
        if not candidate.languages:
            candidate.languages = languages_for(candidate)
        if not candidate.open_to_work_prefs:
            candidate.open_to_work_prefs = prefs_for(candidate)
    return roster



# These four traps stand on the side tables alone, so they are assigned to people already
# in the background rather than adding anyone — the roster and the RNG streams do not move.
#
# Three constraints on who can be picked, all enforced by `validate.py`:
#
# 1. The profile language has to match the posting. Language is what separates the two Rust
#    postings (backend-rust=en, backend-seoul-ko=ko).
# 2. The person has to be findable by that posting's search. ml-platform-tokyo searches
#    `MATCH 'pytorch'`, so the Tokyo traps hold PyTorch but lack Kubernetes practice.
# 3. A control must not become another trap — no open_to_work=false, no missing contact.
TRAP_ASSIGNMENTS: dict[str, dict] = {
    # Nova Vance. en, Seongnam, 11 years of Rust, looking — the strongest on conditions alone.
    "urn:li:person:2hd4v8j0": {
        "trap": "prefs-mismatch",
        "jd": "backend-rust",
        "prefs": [
            OpenToWorkPref(
                desired_title="Backend Engineer",
                location_type="Remote",
                desired_location="Remote",
                start_date="immediately",
                employment_type="Full-time",
            )
        ],
    },
    # Reese Whitlock. The same, but the desired arrangement fits the posting.
    "urn:li:person:e8mk2wpb": {
        "control_for": "prefs-mismatch",
        "jd": "backend-rust",
        "prefs": [
            OpenToWorkPref(
                desired_title="Backend Engineer",
                location_type="Hybrid",
                desired_location="Seoul",
                start_date="immediately",
                employment_type="Full-time",
            )
        ],
    },
    # 예은 구. A Korean profile listing English as native. Writing English on the strength
    # of that entry is abandoning the stated criterion.
    "urn:li:person:q8jw3rvh": {
        "trap": "language-overread",
        "jd": "backend-seoul-ko",
        "languages": [
            Language(name="Korean", proficiency="NATIVE_OR_BILINGUAL"),
            Language(name="English", proficiency="NATIVE_OR_BILINGUAL"),
        ],
    },
    # 시우 도. The same Korean profile with no English entry to be swayed by.
    "urn:li:person:w9td4bmq": {
        "control_for": "language-overread",
        "jd": "backend-seoul-ko",
        "languages": [Language(name="Korean", proficiency="NATIVE_OR_BILINGUAL")],
    },
    # Ellis Dunmore. Tokyo, PyTorch, a master's in ML — and no Kubernetes operations in any
    # position description.
    "urn:li:person:646vfj9a": {
        "trap": "degree-without-practice",
        "jd": "ml-platform-tokyo",
        "educations": [
            Education(
                school_name="Lanternwood University",
                degree_name="Bachelor of Science",
                field_of_study="Statistics",
                start_year=2012,
                end_year=2016,
            ),
            Education(
                school_name="Lanternwood University",
                degree_name="Master of Science",
                field_of_study="Machine Learning",
                start_year=2016,
                end_year=2018,
            ),
        ],
    },
    # Lane Ashby. The same master's, with Kubernetes practice behind it.
    "urn:li:person:c5nq8jtv": {
        "control_for": "degree-without-practice",
        "jd": "ml-platform-tokyo",
        "educations": [
            Education(
                school_name="Cresthollow Institute of Technology",
                degree_name="Bachelor of Science",
                field_of_study="Mathematics",
                start_year=2013,
                end_year=2017,
            ),
            Education(
                school_name="Cresthollow Institute of Technology",
                degree_name="Master of Science",
                field_of_study="Machine Learning",
                start_year=2017,
                end_year=2019,
            ),
        ],
    },
    # Kai Fairbanks. PyTorch and a CKA, but has never run Kubernetes.
    "urn:li:person:fzgfv3y7": {
        "trap": "cert-without-practice",
        "jd": "ml-platform-tokyo",
        "certifications": [
            Certification(
                name="Certified Kubernetes Administrator",
                authority="Cloud Native Computing Foundation",
            )
        ],
    },
    # Blake Merrick. The same certificate, with the operations behind it.
    "urn:li:person:s9wk4bpr": {
        "control_for": "cert-without-practice",
        "jd": "ml-platform-tokyo",
        "certifications": [
            Certification(
                name="Certified Kubernetes Administrator",
                authority="Cloud Native Computing Foundation",
            )
        ],
    },
}


JAPANESE_RATE = 0.55

_JA_TITLE = {
    "Backend": "バックエンドエンジニア",
    "Frontend": "フロントエンドエンジニア",
    "ML": "機械学習エンジニア",
    "Data": "データエンジニア",
    "Infrastructure": "インフラエンジニア",
    "Mobile": "モバイルエンジニア",
}
_JA_SENIORITY = {"Senior": "シニア", "Staff": "スタッフ", "Lead": "リード", "Principal": "プリンシパル"}

# The contact note follows the candidate's language, as `gen.py` does for ko/en.
_JA_CONTACT_NOTE = {"inmail": "InMail 受信可能", "referral": "元同僚からの紹介可能"}


def japanize(roster: list[Candidate]) -> list[Candidate]:
    """Turns some Japan residents into Japanese names and `ja` profiles.

    There was a Tokyo posting and not one Japanese profile in the pool. The share matches
    Korea's (244 of 426).

    Traps and controls are left alone — `same-name` and `duplicate-profile` depend on the
    names. `fill()` has not run yet, so the assigned people carry no `_trap`; without
    blocking their ids here, two Tokyo traps would be renamed out of their setup.
    """
    for candidate in roster:
        if candidate.country != "JP":
            continue
        if getattr(candidate, "_trap", None) or getattr(candidate, "_control_for", None):
            continue
        if getattr(candidate, "_verdict", None) or candidate.id in TRAP_ASSIGNMENTS:
            continue
        rng = seeded(f"profile:japanize:{candidate.id}")
        if rng.random() >= JAPANESE_RATE:
            continue

        candidate.profile_language = "ja"
        candidate.first_name, candidate.last_name = person(rng, "ja")
        title = _JA_TITLE.get(candidate.job_function, "ソフトウェアエンジニア")
        prefix = _JA_SENIORITY.get(candidate.seniority, "")
        skills = "、".join(s.name for s in candidate.skills[:3])
        candidate.headline = f"{prefix}{title} · {skills}" if skills else f"{prefix}{title}"

        # The contact note and the titles move too, or the languages split inside one
        # profile. Titles revert to the canonical English rather than gaining Japanese
        # variants, which en and ko people would then draw from.
        for contact in candidate.contacts:
            note = _JA_CONTACT_NOTE.get(contact.method)
            if note:
                contact.note = note
        for position in candidate.positions:
            position.title = variants.canonical_title(position.title)
    return roster


# All nine qualified for ml-platform-tokyo had English profiles, because they all sat in
# the hand-written 65 that `japanize` leaves alone. That rule is right, so this is solved
# from the other side: give two Japanese ML engineers in the background the one skill they
# lack. A Tokyo ML pool where everyone qualified writes in English is the less realistic
# one.
TOKYO_SKILL_GRANTS: dict[str, str] = {
    "urn:li:person:2vqtrqt9": "MLOps",      # 陽菜 山田, 9.0y. Has PyTorch and Kubernetes
    "urn:li:person:x5w1f5gp": "Kubernetes",  # 莉子 山本, 6.0y. Has PyTorch and MLOps
}


def grant_tokyo_skills(roster: list[Candidate]) -> list[Candidate]:
    """A skill used to judge eligibility goes in under its canonical name, not a variant.

    `check_must_haves` compares with `s.name IN (...)`, so `Kubernetes (K8s)` would be in
    the data and not counted.
    """
    for candidate in roster:
        want = TOKYO_SKILL_GRANTS.get(candidate.id)
        if not want or any(s.name == want for s in candidate.skills):
            continue
        rng = seeded(f"profile:skill_grant:{candidate.id}")
        candidate.skills.append(Skill(want, rng.randint(2, 60)))
    return roster
