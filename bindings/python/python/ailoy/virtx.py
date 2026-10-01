"""virtx, as built into ailoy: the console an agent's tools run in, and what it sees.

Each extension module links its own copy of virtx, so these are distinct types from
``virtx``'s: a ``ConsoleClient`` built by ``virtx`` cannot be handed to an
``AgentBuilder``.
"""

from enum import IntEnum

from . import _ailoy
from ._ailoy import (
    BuildImageResult,
    ConsoleBroken,
    ConsoleClient,
    ConsoleClientBuilder,
    ConsoleRefused,
    VirtxError,
    Directory,
    ExecResult,
    ImageClient,
    ImageEntry,
    ImageSource,
    ReadResult,
    Recipe,
    Step,
)

# The numbers a `ConsoleRefused.code` may hold, named as virtx names them.
ErrorCode = IntEnum("ErrorCode", _ailoy.ERROR_CODES)

__all__ = [
    "BuildImageResult",
    "ConsoleBroken",
    "ConsoleClient",
    "ConsoleClientBuilder",
    "ConsoleRefused",
    "VirtxError",
    "Directory",
    "ErrorCode",
    "ExecResult",
    "ImageClient",
    "ImageEntry",
    "ImageSource",
    "ReadResult",
    "Recipe",
    "Step",
]

# Present only when the extension was built with the `mount` feature, which is the default.
if hasattr(_ailoy, "HostMount"):
    from ._ailoy import HostMount

    __all__.append("HostMount")
