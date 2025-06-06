if __doc__ is None:
    try:
        import importlib.metadata

        meta = importlib.metadata.metadata("ailoy-py")
        __doc__ = meta.get("Description")
    except importlib.metadata.PackageNotFoundError:
        pass

if __doc__ is None:
    import os.path

    if os.path.isfile("README.md"):
        with open("README.md", "r") as f:
            __doc__ = f.read()
    elif os.path.isfile("../README.md"):
        with open("../README.md", "r") as f:
            __doc__ = f.read()
    else:
        __doc__ = "# ailoy-py\n\nPython binding for Ailoy runtime APIs"

from .agent import Agent  # noqa: F401
from .runtime import AsyncRuntime, Runtime  # noqa: F401
from .vector_store import VectorStore  # noqa: F401
