from os import getenv
from typing import Literal

LANGUAGE: Literal["en", "ko"] = getenv("ROGUEWRITE_LANGUAGE") or "en"

if LANGUAGE == "ko":
    from settings_ko import *  # noqa: F403
else:
    from settings_en import *  # noqa: F403
