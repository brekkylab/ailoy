"""Span merging. Get these numbers wrong and trap 3 does not hold."""

from common.dates import AS_OF, merge_spans, months, naive_sum


def span(start: tuple[int, int], end: tuple[int, int] | None) -> tuple[int, int]:
    """A (year, month) pair as an absolute month span. `end=None` means current."""
    s = months(*start)
    e = months(*end) if end else months(*AS_OF)
    return (s, e)


def test_as_of_is_what_the_views_use():
    # views.sql uses 2026*12+8. A mismatch makes the key and the view disagree silently.
    assert AS_OF == (2026, 8)
    assert months(*AS_OF) == 2026 * 12 + 8


def test_a_fully_contained_overlap_is_counted_once():
    # One span fully inside the other.
    spans = [span((2019, 6), (2023, 6)), span((2020, 9), (2022, 9))]
    assert naive_sum(spans) == 72  # 48 + 24
    assert merge_spans(spans) == 48  # 4.0년


def test_spans_that_do_not_touch_are_summed():
    # No overlap, so the sum is the answer. 94 rather than 96 because the spans are
    # half-open — what the trap needs is the relation, not the rounding.
    spans = [span((2016, 1), (2019, 12)), span((2020, 1), (2023, 12))]
    assert naive_sum(spans) == merge_spans(spans) == 94


def test_the_rank_inversion_pair_actually_inverts():
    # The rank-inversion pair.
    a = [span((2018, 3), (2023, 3)), span((2019, 6), (2022, 6))]
    b = [span((2019, 1), (2021, 7)), span((2021, 9), None)]

    assert naive_sum(a) == 96 and merge_spans(a) == 60
    assert naive_sum(b) == 89 and merge_spans(b) == 89

    # A leads on the naive sum; B leads once merged.
    assert naive_sum(a) > naive_sum(b)
    assert merge_spans(a) < merge_spans(b)

    # Both clear the four-year must-have, so no filter separates them.
    assert merge_spans(a) >= 48 and merge_spans(b) >= 48


def test_a_current_position_runs_to_as_of():
    spans = [span((2025, 8), None)]
    assert merge_spans(spans) == 12


def test_partial_overlap_merges_to_the_union():
    spans = [span((2020, 1), (2022, 1)), span((2021, 1), (2023, 1))]
    assert naive_sum(spans) == 48
    assert merge_spans(spans) == 36


def test_three_spans_chaining_into_one_island():
    spans = [
        span((2018, 1), (2020, 1)),
        span((2019, 6), (2021, 6)),
        span((2021, 1), (2023, 1)),
    ]
    assert merge_spans(spans) == 60  # 2018-01 ~ 2023-01
