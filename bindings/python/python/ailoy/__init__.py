"""Agents that drive a language model through tool-augmented turns.

An ``Agent`` is built with an ``AgentBuilder`` and awaited; a turn is ``agent.run(query)``,
iterated with ``async for``. A model is named ``<org>/<name>``
(``"anthropic/claude-sonnet-5-5"``) and runs on the backend set for its org with
``set_lang_model_provider``: by default its CLI, or its own API when it has none.
``LangModel.create(desc)`` builds one to run without an agent. A tool is built by name through
the provider set for it (``set_tool_provider``, ``register_tool``, ``register_mcp_stdio``, ...);
every built-in tool has one from the start.

Messages, specs and tool descriptions are dicts in their JSON form; ``ailoy.types`` spells
out their shapes.

Tools run commands in a console from the ``virtx`` package, which ailoy depends on: a
``virtx.ConsoleClient`` handed to ``AgentBuilder.console`` is shared with the agent.

See the Rust crate's documentation for details.
"""

from . import types
from ._ailoy import (
    Agent,
    AgentBuilder,
    AgentRun,
    AiloyError,
    LangModel,
    register_a2a,
    register_mcp_stdio,
    register_mcp_streamable_http,
    register_tool,
    remove_lang_model_provider,
    remove_tool_provider,
    set_lang_model_provider,
    set_tool_provider,
    unregister_mcp,
)

__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentRun",
    "AiloyError",
    "LangModel",
    "register_a2a",
    "register_mcp_stdio",
    "register_mcp_streamable_http",
    "register_tool",
    "remove_lang_model_provider",
    "remove_tool_provider",
    "set_lang_model_provider",
    "set_tool_provider",
    "types",
    "unregister_mcp",
]
