"""The background 535, and assembling all 600.

**The background is not made weak.** These people are good at their own work and fall
short only on the axis a posting asks about — an eight-year frontend engineer misses a
Rust backend posting because of what they have done, not how well.

The distinction matters because of elimination. Make the background visibly inferior and
an agent can find the core by discarding whatever looks less plausible, getting the right
answer for the wrong reason.
"""

import calendar
from pathlib import Path

from common.dates import AS_OF, months
from common.names import company, person
from common.rng import seeded
from common.writer import dump
from headhunter import variants
from headhunter.fixtures import core
from headhunter.schema import (
    COMPANY_SIZES,
    EMPLOYMENT_TYPES,
    SENIORITY,
    Candidate,
    Contact,
    Position,
    Skill,
)

# Backend is the largest share but not a majority — around 40 Rust people in 600 is what
# makes a search worth running.
JOB_FUNCTIONS: tuple[tuple[str, int], ...] = (
    ("Backend", 30),
    ("Frontend", 20),
    ("ML", 15),
    ("Data", 13),
    ("Infrastructure", 12),
    ("Mobile", 10),
)

# (skill, weight) per function.
#
# Solidity appears nowhere on purpose: the blockchain posting is meant to qualify nobody.
#
# **Skills cross function boundaries.** Pools that were exclusive per function left
# `WHERE job_function='ML' AND skill='Kubernetes'` singling out 12 core people, because
# only the core held that combination. Every combination the core uses is in a pool, at a
# low weight — held by some of that function, not its mainstay.
SKILLS_BY_FUNCTION: dict[str, tuple[tuple[str, int], ...]] = {
    "Backend": (
        ("Python", 25), ("Java", 22), ("Go", 18), ("PostgreSQL", 15),
        ("Kubernetes", 12), ("Distributed Systems", 10), ("Kafka", 8), ("Rust", 6),
    ),
    "Frontend": (
        ("TypeScript", 30), ("React", 25), ("CSS", 18), ("Vue", 12),
        ("Accessibility", 9), ("WebAssembly", 6),
    ),
    "ML": (
        ("Python", 28), ("PyTorch", 24), ("Transformers", 18), ("MLOps", 14),
        # Central to ML platform work. 12 core people hold it, so the background must be able to.
        ("Kubernetes", 16), ("TensorFlow", 10), ("CUDA", 6), ("SQL", 5),
        ("Snowflake", 3),
    ),
    "Data": (
        ("SQL", 30), ("Spark", 22), ("Airflow", 18), ("Python", 16), ("dbt", 14),
        ("Kafka", 10), ("Snowflake", 6), ("Kubernetes", 4), ("PyTorch", 3),
    ),
    "Infrastructure": (
        ("Linux", 25), ("Terraform", 22), ("AWS", 20), ("Kubernetes", 18),
        ("Observability", 9), ("Networking", 6), ("Rust", 4),
    ),
    "Mobile": (
        ("Swift", 25), ("Kotlin", 24), ("iOS", 20), ("Android", 18),
        ("Flutter", 8), ("React Native", 5),
    ),
}


def skills_outside_the_pools() -> list[tuple[str, str]]:
    """(function, skill) pairs the core uses that no pool holds. Must be empty.

    A pair only the core can hold makes `WHERE job_function=… AND skill=…` single them
    out. It lives here rather than in `validate.py` because its input is code, so the
    answer is available before any data exists.
    """
    return sorted(
        {
            (c.job_function, s.name)
            for c in core()
            for s in c.skills
            if s.name not in {n for n, _ in SKILLS_BY_FUNCTION.get(c.job_function, ())}
        }
    )


# Right-skewed: most people are junior.
TENURE_BANDS: tuple[tuple[tuple[int, int], int], ...] = (
    ((1, 3), 35),
    ((4, 8), 40),
    ((9, 15), 25),
)

# **`seeded(name)` returns the same stream every call.** Called inside a loop it makes all
# 535 people identical — and the mistake is silent: the data generates, validation passes,
# and only the distribution shows it. Build each stream once, outside the loop.
LOCATIONS: tuple[tuple[tuple[str, str, str], int], ...] = (
    (("Seoul", "KR", "Seoul, KR"), 55),
    (("Seongnam", "KR", "Seoul, KR"), 15),
    (("Tokyo", "JP", "Tokyo, JP"), 12),
    (("Berlin", "DE", "Berlin, DE"), 9),
    (("San Francisco", "US", "San Francisco, US"), 9),
)

# Drawn from the country, not independently — a Berlin resident rarely writes a Korean
# profile, and the prose layer would try to make sense of one who did.
#
# This is also what splits the two Rust postings. They share the needle `rust` and differ
# only by language, so raising the ko share grows one match set and shrinks the other.
PROFILE_LANGUAGE_BY_COUNTRY: dict[str, tuple[tuple[str, int], ...]] = {
    "KR": (("ko", 60), ("en", 40)),
    "JP": (("en", 100),),
    "DE": (("en", 100),),
    "US": (("en", 100),),
}

# The same four the core uses, in roughly the same proportions. A different set would let
# `WHERE industry NOT IN (...)` single the core out.
#
# Drawn independently of function — an ML engineer at a financial firm is unremarkable, and
# a correlation would let `job_function` predict `industry`.
INDUSTRIES: tuple[tuple[str, int], ...] = (
    ("Computer Software", 45),
    ("Internet", 25),
    ("Information Technology and Services", 20),
    ("Financial Services", 10),
)

# Only the backend titles are registered in `variants.py`, so only they drift. The others
# come back unchanged, which is not an error.
TITLE_BASE: dict[str, str] = {
    "Backend": "Backend Engineer",
    "Frontend": "Frontend Engineer",
    "ML": "ML Engineer",
    "Data": "Data Engineer",
    "Infrastructure": "Infrastructure Engineer",
    "Mobile": "Mobile Engineer",
}

# Against the right-skewed tenure, this puts most people near Senior.
SENIORITY_BY_BAND: dict[tuple[int, int], tuple[tuple[str, int], ...]] = {
    (1, 3): (("Entry", 70), ("Senior", 30)),
    (4, 8): (("Senior", 55), ("Manager", 25), ("Entry", 10), ("Director", 10)),
    (9, 15): (
        ("Senior", 30), ("Manager", 28), ("Director", 22),
        ("VP", 12), ("CXO", 6), ("Owner", 2),
    ),
}
assert set(SENIORITY_BY_BAND) == {band for band, _ in TENURE_BANDS}
assert {name for weights in SENIORITY_BY_BAND.values() for name, _ in weights} <= set(SENIORITY)

# The background reaches the extremes rarely; trap 13 needs the smallest band to be rare.
COMPANY_SIZE_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("1-10", 4), ("11-50", 10), ("51-200", 18), ("201-500", 20),
    ("501-1,000", 16), ("1,001-5,000", 16), ("5,001-10,000", 9), ("10,001+", 7),
)
assert {name for name, _ in COMPANY_SIZE_WEIGHTS} == set(COMPANY_SIZES)

WORKPLACE_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("On-site", 55), ("Hybrid", 30), ("Remote", 15),
)

# All FULL_TIME would itself be a fingerprint: the overlapping-tenure trap and the
# inversion pair use CONTRACT, so with none in the background
# `WHERE employment_type != 'FULL_TIME'` would name them exactly. Non-full-time totals 12%.
#
# No overlap leaks: `_build_positions` lays spans out sequentially, so whatever type is
# drawn, the background has no concurrent employment.
EMPLOYMENT_TYPE_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("FULL_TIME", 88), ("CONTRACT", 7), ("PART_TIME", 3), ("INTERNSHIP", 2),
)
assert {name for name, _ in EMPLOYMENT_TYPE_WEIGHTS} == set(EMPLOYMENT_TYPES)

# INTERNSHIP has to depend on length and position. Drawn independently it produces a
# 104-month Manager internship, or a current internship after a Director role. Two rules:
# not the most recent position, and no longer than 12 months. CONTRACT and PART_TIME are
# left alone — long ones are realistic.
_MAX_INTERNSHIP_MONTHS = 12

# **Drawing then filtering kills the distribution.** Applying 2% across all positions and
# demoting the ineligible ones left a single internship in 1006 positions. Qualify first,
# then weight within the qualifying set.
#
# For ineligible positions: the table above without INTERNSHIP, same relative ratios.
_INELIGIBLE_EMPLOYMENT_TYPE_WEIGHTS: tuple[tuple[str, int], ...] = tuple(
    (name, weight) for name, weight in EMPLOYMENT_TYPE_WEIGHTS if name != "INTERNSHIP"
)

# For qualifying positions only. That set is under 5% of the corpus, so 25% here barely
# moves the overall CONTRACT/PART_TIME shares.
_ELIGIBLE_EMPLOYMENT_TYPE_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("FULL_TIME", 45), ("INTERNSHIP", 35), ("CONTRACT", 12), ("PART_TIME", 8),
)
assert {name for name, _ in _ELIGIBLE_EMPLOYMENT_TYPE_WEIGHTS} == set(EMPLOYMENT_TYPES)

# One shape would give every background headline exactly one comma, and
# `WHERE headline NOT LIKE '%,%'` would leave only the hand-written core. All four shapes
# carry the skill tokens through — the headline is a search surface.
HEADLINE_TEMPLATE_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("join2", 50),     # "{title} · {s1}, {s2}"
    ("modifier", 15),  # "{s1} {title} · {s2}"
    ("and", 15),       # "{title} · {s1} and {s2}"
    ("join3", 20),     # "{title} · {s1}, {s2}, {s3}"
)

# Shorter careers change jobs less.
NUM_POSITIONS_WEIGHTS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (1, 3): ((1, 100),),
    (4, 8): ((1, 30), (2, 70)),
    (9, 15): ((2, 55), (3, 45)),
}

MIN_POSITION_MONTHS = 6
CURRENTLY_EMPLOYED_PROB = 0.8
_AS_OF_MONTHS = months(*AS_OF)
_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

_INMAIL_EN = Contact("inmail", "InMail open")
_INMAIL_KO = Contact("inmail", "InMail 수신 가능")
_REFERRAL_EN = Contact("referral", "Referred by a former colleague")
_REFERRAL_KO = Contact("referral", "전 동료 소개 가능")


def _weighted_choice(rng, table):
    """One value from `((value, weight), …)`, weighted."""
    values = [v for v, _ in table]
    weights = [w for _, w in table]
    return rng.choices(values, weights=weights, k=1)[0]


def _weighted_sample_without_replacement(rng, table, k):
    """`k` distinct values, reweighting over what is left at each step."""
    pool = list(table)
    k = min(k, len(pool))
    chosen = []
    for _ in range(k):
        total = sum(w for _, w in pool)
        pick = rng.uniform(0, total)
        upto = 0.0
        for i, (value, weight) in enumerate(pool):
            upto += weight
            if upto >= pick:
                chosen.append(value)
                pool.pop(i)
                break
    return chosen


def _from_months(total: int) -> tuple[int, int]:
    """The inverse of `common.dates.months`."""
    year, month = divmod(total - 1, 12)
    return year, month + 1


def _segment_lengths(rng, total: int, n: int) -> list[int]:
    """Splits `total` months into `n` segments of at least `MIN_POSITION_MONTHS`.

    With no room to split, falls back to one segment rather than producing a zero- or
    negative-length position.
    """
    if n <= 1 or total < MIN_POSITION_MONTHS * n:
        return [total]
    remaining = total
    lengths = []
    for i in range(n - 1):
        slots_left = n - 1 - i
        max_len = remaining - MIN_POSITION_MONTHS * slots_left
        length = rng.randint(MIN_POSITION_MONTHS, max_len)
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)
    return lengths


def _ident(rng, taken: set[str]) -> str:
    """An eight-character opaque id, the same shape the core uses."""
    while True:
        candidate = "".join(rng.choices(_ID_ALPHABET, k=8))
        if candidate not in taken:
            taken.add(candidate)
            return candidate


def _company_size_cache() -> dict[str, str]:
    """urn → company_size, seeded from the core's hand assignments.

    The core is authoritative: the `inflated-title` trap rests on one urn being "1-10", so
    wherever that urn reappears in the background it has to keep that size.

    Two core people can land on one urn by chance — the company name pool is finite. It
    has happened once. A collision touching a trap or control raises; two judgment-only
    people keep whichever `core()` yields first, which the trap logic does not care about.
    """
    cache: dict[str, str] = {}
    for candidate in core():
        for position in candidate.positions:
            prior = cache.get(position.company_urn)
            if prior is not None and prior != position.company_size:
                trapped = getattr(candidate, "_trap", None) or getattr(
                    candidate, "_control_for", None
                )
                if trapped:
                    raise AssertionError(
                        f"{position.company_urn} has two company_size values inside the "
                        f"core, and one of them is a trap or control ({trapped}): "
                        f"{prior!r} vs {position.company_size!r}"
                    )
                continue  # a collision no trap depends on; keep the first value seen.
            cache[position.company_urn] = position.company_size
    return cache


def _headline(rng, title: str, skills: list[str]) -> str:
    """Varies how skills attach to the title. The skill strings pass through unchanged.

    `join3` needs three skills, so a two-skill profile draws from the other three shapes;
    `_weighted_choice` renormalizes over what is left.
    """
    weights = HEADLINE_TEMPLATE_WEIGHTS
    if len(skills) < 3:
        weights = tuple((k, w) for k, w in weights if k != "join3")
    kind = _weighted_choice(rng, weights)
    s1, s2 = skills[0], skills[1]
    if kind == "modifier":
        return f"{s1} {title} · {s2}"
    if kind == "and":
        return f"{title} · {s1} and {s2}"
    if kind == "join3":
        return f"{title} · {s1}, {s2}, {skills[2]}"
    return f"{title} · {s1}, {s2}"  # join2


# Only reached when the window is very narrow, and then a uniform draw takes over.
_UPDATED_REDRAW_ATTEMPTS = 50


def _draw_updated(rng, latest_start: int) -> tuple[int, int]:
    """Draws `last_updated_at` inside `[latest_start, AS_OF]`.

    **Redrawn, not clamped.** Pushing out-of-window values up to the bound piles every one
    of them onto the same month. After `_UPDATED_REDRAW_ATTEMPTS` misses it falls back to
    a uniform draw inside the window, which always succeeds.
    """
    for _ in range(_UPDATED_REDRAW_ATTEMPTS):
        year = rng.choices([2024, 2025, 2026], weights=[15, 35, 50], k=1)[0]
        month = rng.randint(1, 12 if year != 2026 else 8)
        if months(year, month) >= latest_start:
            return year, month
    return _from_months(rng.randint(latest_start, _AS_OF_MONTHS))


def _employment_type(rng, length: int, is_last: bool, missing_end: bool) -> str:
    """The employment type for one position.

    When it qualifies (not the most recent, no longer than `_MAX_INTERNSHIP_MONTHS`, and an
    `end_year` actually shown)
    it draws from a different table — it is not drawn and then filtered. Filtering would
    multiply a narrow condition (under 5% of positions) by a low weight (2%) and leave
    almost nobody; measured, one internship survived out of 47 eligible positions. Either
    way there is one `_weighted_choice` call, so the rng is consumed identically.

    `missing_end` is part of the condition because the weak trap that clears `end_year`
    leaves this position looking current whatever its real length. A long CONTRACT or
    PART_TIME still reads as realistic, but
    A years-long open-ended internship is a contradiction whatever its real length, so such
    a position is excluded from the INTERNSHIP pool entirely.
    """
    eligible = not is_last and not missing_end and length <= _MAX_INTERNSHIP_MONTHS
    weights = _ELIGIBLE_EMPLOYMENT_TYPE_WEIGHTS if eligible else _INELIGIBLE_EMPLOYMENT_TYPE_WEIGHTS
    return _weighted_choice(rng, weights)


def _build_positions(
    *,
    rng_company,
    rng_company_variant,
    rng_location_variant,
    rng_size,
    rng_employment,
    rng_workplace,
    rng_employed,
    rng_segments,
    rng_gap,
    title: str,
    location_canonical: str,
    total_months: int,
    num_positions: int,
    force_missing_end_year: bool,
    force_short_gap: bool,
    company_size_by_urn: dict[str, str],
) -> list[Position]:
    """One person's career, oldest first.

    `force_missing_end_year` leaves a past role looking current — a weak trap axis.
    `force_short_gap` opens a 1–4 month gap, small enough not to reach the
    `employment-gap` trap's 12 months but enough to make results untidy.

    `company_size_by_urn` pins one size per urn. Several people work at the same company,
    and redrawing the size per position would make it differ profile to profile — which
    breaks the `inflated-title` trap's premise that a given urn is "1-10".
    """
    currently_employed = rng_employed.random() < CURRENTLY_EMPLOYED_PROB
    trailing_gap = 0 if currently_employed else rng_employed.randint(1, 12)

    lengths = _segment_lengths(rng_segments, total_months, num_positions)
    n = len(lengths)

    # The gap is inserted between segments and the start pushed back by the same amount, so
    # the merged tenure still matches the planned `total_months`.
    gap_before = [0] * n
    if force_short_gap and n >= 2:
        gap_index = rng_gap.randint(1, n - 1)
        gap_before[gap_index] = rng_gap.randint(1, 4)

    span_total = sum(lengths) + sum(gap_before) + trailing_gap
    cursor = _AS_OF_MONTHS - span_total

    positions = []
    for i, length in enumerate(lengths):
        cursor += gap_before[i]
        start_abs = cursor
        end_abs = start_abs + length
        cursor = end_abs

        is_last = i == n - 1
        current = is_last and currently_employed
        missing_end = force_missing_end_year and not is_last and i == 0 and n >= 2

        name, urn = company(rng_company)
        display_name = variants.pick(rng_company_variant, variants.company_variants(name))
        display_location = variants.pick(
            rng_location_variant, variants.location_variants(location_canonical)
        )
        start = _from_months(start_abs)
        end = None if (current or missing_end) else _from_months(end_abs)

        size = company_size_by_urn.get(urn)
        if size is None:
            size = _weighted_choice(rng_size, COMPANY_SIZE_WEIGHTS)
            company_size_by_urn[urn] = size

        positions.append(
            Position(
                title=title,
                company_name=display_name,
                company_urn=urn,
                company_size=size,
                employment_type=_employment_type(rng_employment, length, is_last, missing_end),
                workplace_type=_weighted_choice(rng_workplace, WORKPLACE_WEIGHTS),
                location=display_location,
                description="",  # filled by the prose layer
                start_year=start[0],
                start_month=start[1],
                end_year=end[0] if end else None,
                end_month=end[1] if end else None,
            )
        )
    return positions


def background(count: int, weak_trap_count: int) -> list[Candidate]:
    """`count` background people, `weak_trap_count` of them carrying a weak trap.

    Spelling drift applies to everyone — that is realistic variety, not noise. The weak
    traps are the two structural ones: a past role left looking current, and a short gap
    between adjacent positions. Both muddy a computed value without being large enough to
    flip a judgment.

    One stream per axis, built outside the loop.
    """
    rng_job = seeded("background:job_function")
    rng_band = seeded("background:tenure_band")
    rng_years = seeded("background:tenure_years")
    rng_seniority = seeded("background:seniority")
    rng_location = seeded("background:location")
    rng_language = seeded("background:language")
    rng_industry = seeded("background:industry")
    rng_skill_count = seeded("background:skill_count")
    rng_skill_pick = seeded("background:skill_pick")
    rng_skill_variant = seeded("background:skill_variant")
    rng_title_variant = seeded("background:title_variant")
    rng_endorsement = seeded("background:endorsement")
    rng_name = seeded("background:name")
    rng_ids = seeded("background:ids")
    rng_num_positions = seeded("background:num_positions")
    rng_company = seeded("background:company")
    rng_company_variant = seeded("background:company_variant")
    rng_location_variant = seeded("background:location_variant")
    rng_size = seeded("background:company_size")
    rng_employment = seeded("background:employment_type")
    rng_workplace = seeded("background:workplace")
    rng_headline = seeded("background:headline")
    rng_employed = seeded("background:employed")
    rng_segments = seeded("background:segments")
    rng_gap = seeded("background:gap")
    rng_weak_pick = seeded("background:weak_trap_pick")
    rng_weak_kind = seeded("background:weak_trap_kind")
    rng_open_to_work = seeded("background:open_to_work")
    rng_connections = seeded("background:connections")
    rng_updated = seeded("background:updated_at")
    rng_updated_day = seeded("background:updated_at_day")
    rng_contact = seeded("background:contact")

    taken_idents = {c.id.rsplit(":", 1)[-1] for c in core()}
    company_size_by_urn = _company_size_cache()
    weak_trap_indices = set(rng_weak_pick.sample(range(count), weak_trap_count))

    roster: list[Candidate] = []
    for i in range(count):
        job_function = _weighted_choice(rng_job, JOB_FUNCTIONS)
        band = _weighted_choice(rng_band, TENURE_BANDS)
        years = rng_years.randint(*band)
        seniority = _weighted_choice(rng_seniority, SENIORITY_BY_BAND[band])

        city, country, location_canonical = _weighted_choice(rng_location, LOCATIONS)
        language = _weighted_choice(rng_language, PROFILE_LANGUAGE_BY_COUNTRY[country])
        industry = _weighted_choice(rng_industry, INDUSTRIES)

        # Floor of 2, not 3: nine core people have exactly two skills, and a floor of 3
        # made `HAVING COUNT(*) = 2` name them. The generated range has to contain the
        # hand-written one.
        k = rng_skill_count.choices([2, 3, 4, 5, 6], weights=[10, 20, 35, 30, 15], k=1)[0]
        chosen_skills = _weighted_sample_without_replacement(
            rng_skill_pick, SKILLS_BY_FUNCTION[job_function], k
        )
        skill_display = [
            variants.pick(rng_skill_variant, variants.skill_variants(name))
            for name in chosen_skills
        ]
        skills = [
            Skill(display, rng_endorsement.randint(2, 60)) for display in skill_display
        ]

        is_weak = i in weak_trap_indices
        num_positions = _weighted_choice(rng_num_positions, NUM_POSITIONS_WEIGHTS[band])
        if is_weak:
            num_positions = max(num_positions, 2)
        weak_kind = rng_weak_kind.choice(["missing_end_year", "short_gap"]) if is_weak else None

        base_title = TITLE_BASE[job_function]
        canonical_title = f"Senior {base_title}" if seniority not in ("Entry",) else base_title
        title = variants.pick(rng_title_variant, variants.title_variants(canonical_title))

        positions = _build_positions(
            rng_company=rng_company,
            rng_company_variant=rng_company_variant,
            rng_location_variant=rng_location_variant,
            rng_size=rng_size,
            rng_employment=rng_employment,
            rng_workplace=rng_workplace,
            rng_employed=rng_employed,
            rng_segments=rng_segments,
            rng_gap=rng_gap,
            title=title,
            location_canonical=location_canonical,
            total_months=years * 12,
            num_positions=num_positions,
            force_missing_end_year=is_weak and weak_kind == "missing_end_year",
            force_short_gap=is_weak and weak_kind == "short_gap",
            company_size_by_urn=company_size_by_urn,
        )

        locale = "ko" if language == "ko" else "en"
        first, last = person(rng_name, locale)
        ident = _ident(rng_ids, taken_idents)

        contacts: list[Contact] = []
        if rng_contact.random() < 0.25:
            method_pool = (
                (_INMAIL_KO, _REFERRAL_KO) if language == "ko" else (_INMAIL_EN, _REFERRAL_EN)
            )
            contacts = [rng_contact.choice(method_pool)]

        headline = _headline(rng_headline, title, skill_display) if skill_display else title
        # The update cannot precede the latest position's start, or the profile records a
        # job it had not yet been written to hold.
        latest_start = months(positions[-1].start_year, positions[-1].start_month)
        updated_year, updated_month = _draw_updated(rng_updated, latest_start)
        # The day is scattered too. Fixed at 01, `WHERE last_updated_at NOT LIKE '%-01'`
        # would pick out the core. Only the year and month affect any tenure arithmetic.
        #
        # **From its own stream.** Drawing it from `rng_updated` would shift every later
        # person's year and month by one position.
        updated_day = rng_updated_day.randint(1, calendar.monthrange(updated_year, updated_month)[1])

        roster.append(
            Candidate(
                id=f"urn:li:person:{ident}",
                first_name=first,
                last_name=last,
                headline=headline,
                city=city,
                country=country,
                industry=industry,
                job_function=job_function,
                seniority=seniority,
                profile_language=language,
                open_to_work=rng_open_to_work.random() < 0.3,
                # Ceiling of 3000. The core reaches 2643, so a ceiling of 2500 left
                # `WHERE connections_count > 2500` naming one person. Headroom, not a fit.
                connections_count=rng_connections.randint(60, 3000),
                last_updated_at=f"{updated_year:04d}-{updated_month:02d}-{updated_day:02d}",
                public_profile_url=f"https://www.linkedin.com/in/{ident}",
                summary="",  # filled by the prose layer
                positions=positions,
                skills=skills,
                contacts=contacts,
            )
        )
    return roster


def assemble() -> list[Candidate]:
    """The core 65 plus the background 535, sorted by id. Side tables are filled after."""
    from headhunter.profile import fill, grant_tokyo_skills, japanize

    roster = sorted(core() + background(535, 40), key=lambda c: c.id)
    # Order matters: `japanize` changes `profile_language` and `fill` reads it. Reversed,
    # a Japanese profile gets an English language list.
    return fill(grant_tokyo_skills(japanize(roster)))


def main() -> None:
    out_path = Path(__file__).resolve().parents[2] / "data" / "candidates.json"
    dump(out_path, [c.to_json() for c in assemble()])


if __name__ == "__main__":
    main()
