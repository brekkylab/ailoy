"""Date arithmetic in whole months.

Profile spans are `{year, month}` with no day, so `year*12+month` carries exactly the
information there is and makes comparison and difference integer work.
"""

# Every "now" in the dataset. **Must equal `2026*12+8` in `views.sql`.** A mismatch makes
# the answer key and the view disagree with no error from either SQL or Python;
# `validate.py::check_the_truth_recomputes` catches it on the answer-key side only.
AS_OF: tuple[int, int] = (2026, 8)


def months(year: int, month: int = 1) -> int:
    return year * 12 + month


def naive_sum(spans: list[tuple[int, int]]) -> int:
    """The wrong answer, on purpose — trap 3 is the gap between this and `merge_spans`."""
    return sum(end - start for start, end in spans)


def merge_spans(spans: list[tuple[int, int]]) -> int:
    """Total months with overlap removed. Must agree with `candidate_tenure` in views.sql."""
    if not spans:
        return 0

    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)
