"""Whether generation is deterministic. The data is committed, so a rerun that produces
a diff makes the committed data untrustworthy."""

import json
from pathlib import Path

from common import names
from common.rng import seeded
from common.writer import dump


def test_the_same_stream_name_gives_the_same_sequence():
    a = [seeded("locations").random() for _ in range(3)]
    b = [seeded("locations").random() for _ in range(3)]
    assert a == b


def test_different_stream_names_are_independent():
    # Captured before the other stream is drained, or the assertion proves nothing.
    before = seeded("locations").random()

    other = seeded("skills")
    for _ in range(100):
        other.random()

    assert seeded("locations").random() == before


def test_names_are_drawn_deterministically():
    assert names.person(seeded("p"), "en") == names.person(seeded("p"), "en")
    assert names.company(seeded("c")) == names.company(seeded("c"))


def test_a_company_urn_is_stable_for_a_name():
    # The spelling drifts across profiles; the urn is the only evidence they are one company.
    rng = seeded("c")
    name, urn = names.company(rng)
    assert urn.startswith("urn:li:organization:")
    assert names.urn_for(name) == urn


def test_legal_suffixes_do_not_change_the_urn():
    """The suffixes the drift adds still point at the same company."""
    base = names.urn_for("Nordwind Systems")
    for suffix in ("Inc.", "Corporation", "Co., Ltd."):
        assert names.urn_for(f"Nordwind Systems {suffix}") == base, suffix


def test_different_companies_get_different_urns():
    """A second word that is not a legal suffix is part of the name.

    Break this and 144 companies collapse into 18 urns, so the duplicate-profile trap
    reads two people at different companies as one.
    """
    assert names.urn_for("Nordwind Systems") != names.urn_for("Nordwind Labs")
    assert names.urn_for("Nordwind") != names.urn_for("Nordwind Systems")


def test_korean_names_are_available():
    first, last = names.person(seeded("k"), "ko")
    # Trap 10 needs these.
    assert any("가" <= ch <= "힣" for ch in first + last)


def test_no_generated_name_is_on_the_denylist():
    for i in range(200):
        first, last = names.person(seeded(f"p{i}"), "en")
        assert f"{first} {last}" not in names.DENYLIST
        company, _ = names.company(seeded(f"c{i}"))
        assert company not in names.DENYLIST


def test_dump_writes_sorted_keys_and_a_trailing_newline(tmp_path: Path):
    path = tmp_path / "out.json"
    dump(path, {"b": 1, "a": {"d": 2, "c": 3}})
    text = path.read_text()
    assert text.endswith("\n")
    # Sorted, not insertion order, so the bytes survive a dict reordering.
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == {"b": 1, "a": {"d": 2, "c": 3}}


def test_dump_is_byte_identical_across_runs(tmp_path: Path):
    data = {"z": [3, 1, 2], "a": {"nested": True}}
    first, second = tmp_path / "1.json", tmp_path / "2.json"
    dump(first, data)
    dump(second, data)
    assert first.read_bytes() == second.read_bytes()
