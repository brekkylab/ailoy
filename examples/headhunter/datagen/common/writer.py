"""JSON output that is byte-stable across runs.

The data is committed, so rerunning the generator must not produce a diff. `sort_keys`
removes dict insertion order; the fixed indent keeps git showing only the keys that
changed, which a single-line JSON cannot.
"""

import json
from pathlib import Path
from typing import Any


def dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
