"""cortex, as built into ailoy: the console an agent's tools run in, and what it sees.

The same classes as the ``cortex-py`` package, but not the same types. Each extension module
links its own copy of cortex, so a ``ConsoleClient`` built by ``cortex-py`` cannot be handed to
an ``AgentBuilder`` — build the console from here.
"""

from enum import IntEnum

from . import _ailoy
from ._ailoy import (
    BuildImageResult,
    ConsoleBroken,
    ConsoleClient,
    ConsoleClientBuilder,
    ConsoleRefused,
    CortexError,
    Directory,
    ExecResult,
    ImageClient,
    ImageEntry,
    ImageSource,
    NetworkAccess,
    ReadResult,
    Recipe,
    Step,
)

# The numbers a `ConsoleRefused.code` may hold, named as cortex names them.
ErrorCode = IntEnum("ErrorCode", _ailoy.ERROR_CODES)

__all__ = [
    "BuildImageResult",
    "ConsoleBroken",
    "ConsoleClient",
    "ConsoleClientBuilder",
    "ConsoleRefused",
    "CortexError",
    "Directory",
    "ErrorCode",
    "ExecResult",
    "ImageClient",
    "ImageEntry",
    "ImageSource",
    "NetworkAccess",
    "ReadResult",
    "Recipe",
    "Step",
]

# Present only when the extension was built with the `mount` feature, which is the default.
if hasattr(_ailoy, "HostMount"):
    from ._ailoy import HostMount

    __all__.append("HostMount")
