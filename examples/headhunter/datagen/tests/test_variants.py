"""Whether the spelling drift holds its FTS5 recall target.

FTS5 finds a word when a token is shared and misses it otherwise. `rust-lang` splits at
the hyphen into `rust` + `lang` and answers `MATCH 'rust'`; `러스트` shares nothing and
does not. That asymmetry is the intended difficulty, and these tests pin how much of it
there is — neither too easy nor impossible.
"""

from common.rng import seeded
from headhunter.variants import (
    canonicals,
    company_variants,
    fts5_recall,
    location_variants,
    needle_for,
    pick,
    skill_variants,
    title_variants,
)


def test_rust_variants_include_the_forms_spec_names():
    variants = skill_variants("Rust")
    for form in ("Rust", "rust-lang", "Rust Lang", "Async Rust", "Rust (Programming Language)"):
        assert form in variants


def test_a_variant_sharing_a_token_is_recalled():
    # The Latin variants share the `rust` token.
    assert fts5_recall("Rust", ["rust-lang", "Rust Lang", "Async Rust"]) == 1.0


def test_korean_variants_are_not_recalled():
    # The intended difficulty: found through the distribution, not the search.
    assert fts5_recall("Rust", ["러스트"]) == 0.0


def test_an_adjacent_skill_is_not_recalled():
    # Tokio is Rust ecosystem and shares no token with it.
    assert fts5_recall("Rust", ["Tokio"]) == 0.0


def test_every_table_lands_in_the_recall_band():
    """**All seven** tables have to sit inside 0.85–0.95.

    Measuring only `Rust` once hid two: `Kubernetes` sat at 0.73 because `K8s` carries
    weight 20 and shares no token, and `Seoul, KR` at 0.45 because the needle was picked
    alphabetically and came out `kr`. Both surface much later, as data that inexplicably
    cannot be searched.

        recall = 1 − (weight not sharing the needle) / (total weight)
    """
    # From `variants.canonicals()`, so adding a canonical does not mean editing two lists.
    # The third element extracts the value from a candidate and is not needed here.
    for canonical, accessor, _field in canonicals():
        rng = seeded(f"recall:{canonical}")
        chosen = [pick(rng, accessor(canonical)) for _ in range(200)]
        recall = fts5_recall(canonical, chosen)
        assert 0.85 <= recall <= 0.95, (
            f"{canonical}: recall {recall:.3f} outside the 0.85-0.95 band "
            f"(needle={needle_for(canonical)!r})"
        )


def test_the_needle_is_the_word_a_searcher_would_use():
    """The needle is written down because no rule derives it.

    In `Senior Backend Engineer` the first token is a modifier and the alphabetical one is
    right; in `Seoul, KR` it is the other way round.
    """
    assert needle_for("Senior Backend Engineer") == "backend"
    assert needle_for("Seoul, KR") == "seoul"
    assert needle_for("Rust") == "rust"
    # Unregistered values fall back to the first token
    assert needle_for("Go") == "go"


def test_company_variants_differ_only_in_suffix():
    variants = company_variants("Finlogic Systems")
    assert "Finlogic Systems" in variants
    # The root survives a suffix, which is what `urn_for` keys on.
    for variant in variants:
        assert variant.split()[0] == "Finlogic"


def test_title_variants_include_a_korean_form():
    variants = title_variants("Senior Backend Engineer")
    assert any(any("가" <= ch <= "힣" for ch in v) for v in variants)


def test_location_variants_include_a_korean_form():
    variants = location_variants("Seoul, KR")
    assert "서울" in variants


def test_picking_is_deterministic():
    a = [pick(seeded("s"), skill_variants("Rust")) for _ in range(5)]
    b = [pick(seeded("s"), skill_variants("Rust")) for _ in range(5)]
    assert a == b
