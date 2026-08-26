"""Deterministic randomness, one stream per name.

Sharing one RNG across the generator means a change that draws one extra number in an
early step shifts every later step's output. The data is committed, so that turns a small
change into a large diff.
"""

import hashlib
import random

# Change this and all 600 people become different people.
ROOT_SEED = "headhunter-2026-08"


def seeded(name: str) -> random.Random:
    """An independent stream for `name`.

    Seeded from SHA-256 rather than `hash()`, which varies per process under
    `PYTHONHASHSEED`.
    """
    digest = hashlib.sha256(f"{ROOT_SEED}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
