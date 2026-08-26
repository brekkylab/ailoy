"""The profile schema, one field per column.

**What is absent is the design.** There is no `email_address` — a sourced candidate
normally has no way to reach them, and `Contact` rows exist only for those who do. And
nothing derived is stored: no total tenure, no current title. Store those and trap 3
(concurrent employment) disappears, because the room to compute is the room to get it
wrong.

`spans_for` and `gap_months` live here because "this candidate's position spans" is a
property of `Candidate`. Both `truth.py` and the fixture tests import them; they each
reimplemented the arithmetic once, and the two definitions drifting apart is silent.
"""

from dataclasses import dataclass, field

from common.dates import AS_OF, months

# Recruiter's Seniority filter, simplified. CXO and Owner have to stay distinct or trap 13
# (the "CTO" of a three-person startup) cannot be expressed.
SENIORITY: tuple[str, ...] = (
    "Entry", "Senior", "Manager", "Director", "VP", "CXO", "Owner",
)

# LinkedIn's own bands. The smallest one includes three people, which trap 13 needs.
COMPANY_SIZES: tuple[str, ...] = (
    "1-10", "11-50", "51-200", "201-500",
    "501-1,000", "1,001-5,000", "5,001-10,000", "10,001+",
)

WORKPLACE_TYPES: tuple[str, ...] = ("On-site", "Remote", "Hybrid")
EMPLOYMENT_TYPES: tuple[str, ...] = ("FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP")


@dataclass
class Position:
    title: str
    company_name: str
    company_urn: str
    company_size: str
    employment_type: str
    workplace_type: str
    location: str
    description: str
    start_year: int
    start_month: int
    # None means current. One value to keep in sync instead of two.
    end_year: int | None = None
    end_month: int | None = None


@dataclass
class Skill:
    name: str
    endorsement_count: int


@dataclass
class Education:
    school_name: str
    degree_name: str
    field_of_study: str
    start_year: int
    end_year: int


@dataclass
class Certification:
    name: str
    authority: str


@dataclass
class Language:
    name: str
    proficiency: str


@dataclass
class OpenToWorkPref:
    """What the candidate entered under Open to Work. Visible to recruiters only."""

    desired_title: str
    location_type: str
    desired_location: str
    start_date: str
    employment_type: str


@dataclass
class Contact:
    """A way to reach someone. Absent by default, which is what trap 12 rests on."""

    method: str  # "inmail" | "referral"
    note: str


@dataclass
class Candidate:
    id: str
    first_name: str
    last_name: str
    headline: str
    city: str
    country: str
    industry: str
    job_function: str
    seniority: str
    profile_language: str
    open_to_work: bool
    connections_count: int
    last_updated_at: str
    public_profile_url: str
    # Filled by the prose layer; empty string until then.
    summary: str = ""
    positions: list[Position] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    educations: list[Education] = field(default_factory=list)
    certifications: list[Certification] = field(default_factory=list)
    languages: list[Language] = field(default_factory=list)
    open_to_work_prefs: list[OpenToWorkPref] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)

    def to_json(self) -> dict:
        """One entry in `candidates.json`.

        Not `dataclasses.asdict`, so that a `None` end date drops the key entirely — a
        current role reads as an absent key, which is the shape the profile has.
        """
        from dataclasses import asdict

        out = asdict(self)
        for position in out["positions"]:
            if position["end_year"] is None:
                position.pop("end_year")
                position.pop("end_month")
        return out


def spans_for(candidate: Candidate) -> list[tuple[int, int]]:
    """Positions as absolute month spans. The one definition `truth.py`, `validate.py`,
    and the fixture tests share.

    A current role fills its end with `AS_OF` — the only way to express "until now" as a
    span. `truth.py`'s `verifiable_months` cuts that fill back at `last_updated_at`.
    """
    out = []
    for position in candidate.positions:
        start = months(position.start_year, position.start_month)
        end = (
            months(position.end_year, position.end_month)
            if position.end_year
            else months(*AS_OF)
        )
        out.append((start, end))
    return out


def gap_months(spans: list[tuple[int, int]]) -> int:
    """The longest gap between positions. What trap 9 rests on.

    Half-open: a role ending the same month the next begins is a gap of zero. Missing that
    boundary once gave a control a gap of 1 instead of 0.
    """
    if len(spans) < 2:
        return 0
    ordered = sorted(spans)
    gaps = [
        later[0] - earlier[1]
        for earlier, later in zip(ordered, ordered[1:])
        if later[0] > earlier[1]
    ]
    return max(gaps, default=0)
