"""Whether the core and the background draw on the same vocabulary.

The input is two modules, not generated data, so this is knowable before generating — and
has to be. `validate.py`'s fingerprint checks read the JSON and do not overlap.
"""

from headhunter.gen import (
    INDUSTRIES,
    LOCATIONS,
    PROFILE_LANGUAGE_BY_COUNTRY,
    skills_outside_the_pools,
)
from headhunter.fixtures import core


def test_the_core_uses_no_skill_outside_its_functions_pool():
    """A skill outside the pools belongs to the core alone, which singles the core out."""
    assert skills_outside_the_pools() == []


def test_the_core_uses_no_industry_outside_the_background_pool():
    names = {n for n, _ in INDUSTRIES}
    assert {c.industry for c in core()} <= names


def test_the_core_uses_no_city_outside_the_background_pool():
    cities = {city for (city, _, _), _ in LOCATIONS}
    assert {c.city for c in core()} <= cities


def test_every_country_the_core_uses_has_a_language_rule():
    """Without a rule, `background()` cannot pick a language for that country."""
    assert {c.country for c in core()} <= set(PROFILE_LANGUAGE_BY_COUNTRY)
