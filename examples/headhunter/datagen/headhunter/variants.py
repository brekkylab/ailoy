"""Spelling drift — what defeats an exact-match query.

Spreading variants evenly would leave the canonical spelling at a sixth of the field and
make search nearly useless. Real profiles cluster on the common spelling. The weights below
also keep FTS5 recall inside 0.85–0.95 by holding the token-disjoint variants (`러스트`,
`Tokio`) to a small share.
"""

import random
from typing import Callable

# Each entry is `(needle, variants)` where a variant is `(spelling, weight)`.
#
# **The needle is written down because it cannot be derived.** In `Senior Backend
# Engineer` the alphabetically first token is `backend` (right) but the first token in the
# string is `senior` (a modifier); in `Seoul, KR` it is the other way round. No rule gets
# both, so a person picks.
#
#     recall = 1 − (weight of variants not sharing the needle) / (total weight)
#
# The values below put all seven tables inside 0.85–0.95. Changing one weight moves only
# that table.
_SKILL: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "Rust": ("rust", (
        ("Rust", 50),
        ("rust-lang", 15),
        ("Rust Lang", 10),
        ("Async Rust", 8),
        ("Rust (Programming Language)", 7),
        # shares no token with the needle — the agent has to find it in the distribution
        ("러스트", 6),
        ("Tokio", 4),
    )),
    "Kubernetes": ("kubernetes", (
        ("Kubernetes", 55),
        ("Kubernetes (K8s)", 20),
        ("kubernetes-admin", 15),
        # common in the wild but shares no token with `kubernetes`, hence the low weight
        ("K8s", 6),
        ("쿠버네티스", 4),
    )),
    "Distributed Systems": ("distributed", (
        ("Distributed Systems", 60),
        ("Distributed Computing", 15),
        ("distributed-systems", 10),
        ("Large-scale Distributed Systems", 8),
        ("분산 시스템", 7),
    )),
}

_TITLE: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "Senior Backend Engineer": ("backend", (
        ("Senior Backend Engineer", 45),
        ("Sr. Backend Engineer", 18),
        ("Backend Engineer II", 12),
        ("Staff Software Engineer, Backend", 10),
        ("서버 개발자", 8),
        ("백엔드 엔지니어", 7),
    )),
    "Backend Engineer": ("backend", (
        ("Backend Engineer", 55),
        ("Software Engineer, Backend", 20),
        ("Backend Developer", 15),
        ("Server Engineer", 6),
        ("백엔드 개발자", 4),
    )),
}

_LOCATION: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "Seoul, KR": ("seoul", (
        ("Seoul, KR", 45),
        ("Greater Seoul Area", 22),
        ("Seoul, South Korea", 20),
        ("서울", 8),
        ("서울, 대한민국", 5),
    )),
    "Tokyo, JP": ("tokyo", (
        ("Tokyo, JP", 55),
        ("Tokyo, Japan", 25),
        ("Greater Tokyo Area", 12),
        ("도쿄", 8),
    )),
}

# Must match `LEGAL_SUFFIXES` in `common/names.py`, which is the source. These leave the
# root untouched, so `urn_for` returns the same urn.
_COMPANY_SUFFIX: tuple[tuple[str, int], ...] = (
    ("", 60),
    (" Inc.", 18),
    (" Corporation", 12),
    (" Co., Ltd.", 10),
)


def skill_variants(canonical: str) -> tuple[str, ...]:
    return _variants(_SKILL, canonical)


def title_variants(canonical: str) -> tuple[str, ...]:
    return _variants(_TITLE, canonical)


def canonical_title(variant: str) -> str:
    """A title variant back to its canonical, which is always English. Unregistered values
    map to themselves. No variant belongs to two canonicals.
    """
    return _CANONICAL_BY_VARIANT.get(variant, variant)


_CANONICAL_BY_VARIANT: dict[str, str] = {
    variant: canonical
    for canonical, (_, variants) in _TITLE.items()
    for variant, _weight in variants
}


def location_variants(canonical: str) -> tuple[str, ...]:
    return _variants(_LOCATION, canonical)


def company_variants(canonical: str) -> tuple[str, ...]:
    return tuple(f"{canonical}{suffix}" for suffix, _ in _COMPANY_SUFFIX)


def _variants(table: dict, canonical: str) -> tuple[str, ...]:
    """Just the spellings. An unregistered canonical is a one-element tuple."""
    entry = table.get(canonical)
    return tuple(v for v, _ in entry[1]) if entry else (canonical,)


def needle_for(canonical: str) -> str:
    """The token to search this canonical by. Searches all three tables so the caller
    need not say which kind it is; no canonical appears in two.
    """
    for table in (_SKILL, _TITLE, _LOCATION):
        entry = table.get(canonical)
        if entry:
            return entry[0]
    tokens = _tokens_in_order(canonical)
    return tokens[0] if tokens else ""


def pick(rng: random.Random, variants: tuple[str, ...]) -> str:
    """Picks one, weighted.

    `variants` has to be what one of the accessors above returned — tuple identity is how
    the weights are found. A hand-built tuple falls back to uniform.
    """
    return rng.choices(list(variants), weights=_weights_for(variants), k=1)[0]


def _weights_for(variants: tuple[str, ...]) -> list[int]:
    """The weights for `variants`, or uniform when the tuple is not from a table."""
    for table in (_SKILL, _TITLE, _LOCATION):
        for _needle, entries in table.values():
            if tuple(v for v, _ in entries) == variants:
                return [w for _, w in entries]
    if variants and all(v.startswith(variants[0]) for v in variants):
        return [w for _, w in _COMPANY_SUFFIX][: len(variants)]
    return [1] * len(variants)


def _tokens_in_order(text: str) -> list[str]:
    """The same split as `_tokens`, in order rather than as a set."""
    out, current = [], []
    for ch in text:
        if ch.isalnum():
            current.append(ch.lower())
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _tokens(text: str) -> set[str]:
    """What FTS5's unicode61 tokenizer does: split on non-alphanumerics, lowercase.

    `rust-lang` becomes `{rust, lang}`; `러스트` stays one token and shares none with
    `rust`, which is the whole point of it being a variant.
    """
    return set(_tokens_in_order(text))


def fts5_recall(canonical: str, chosen: list[str]) -> float:
    """What share of `chosen` a `MATCH needle_for(canonical)` finds. Target: 0.85–0.95.

    Too much drift makes search meaningless and too little leaves no trap; this number is
    the only way to tell which side you are on.
    """
    if not chosen:
        return 0.0
    needle = needle_for(canonical)
    hits = sum(1 for value in chosen if needle in _tokens(value))
    return hits / len(chosen)


def canonicals() -> tuple[tuple[str, Callable, Callable], ...]:
    """`(canonical, accessor, extractor)` for each of the seven tables.

    One list, imported by both `validate.py` and the tests, so the seven are not written
    down twice.
    """
    return (
        ("Rust", skill_variants, lambda c: [s["name"] for s in c["skills"]]),
        ("Kubernetes", skill_variants, lambda c: [s["name"] for s in c["skills"]]),
        ("Distributed Systems", skill_variants, lambda c: [s["name"] for s in c["skills"]]),
        ("Senior Backend Engineer", title_variants, lambda c: [p["title"] for p in c["positions"]]),
        ("Backend Engineer", title_variants, lambda c: [p["title"] for p in c["positions"]]),
        ("Seoul, KR", location_variants, lambda c: [p["location"] for p in c["positions"]]),
        ("Tokyo, JP", location_variants, lambda c: [p["location"] for p in c["positions"]]),
    )
