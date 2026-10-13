"""Agents that drive a language model through tool-augmented turns.

An ``Agent`` is built with an ``AgentBuilder`` and awaited; a turn is ``agent.run(query)``,
iterated with ``async for``. A model is named ``<org>/<name>``
(``"anthropic/claude-sonnet-5-5"``) and runs on the backend set for its org with
``set_lang_model_provider``: by default its CLI, or its own API when it has none.
``LangModel.create(desc)`` builds one to run without an agent. Tools resolve by name from
process-wide registries (``register_tool``, ``register_mcp_stdio``, ...), each starting with a
``"default"`` entry of every built-in tool.

Messages, specs and tool descriptions are dicts in their JSON form; ``ailoy.types`` spells
out their shapes.

Tools run commands in a virtx console from ``ailoy.virtx``, built into this package so an
``Agent`` can share its session.

See the Rust crate's documentation for details.
"""

from . import virtx, types
from ._ailoy import (
    Agent,
    AgentBuilder,
    AgentRun,
    AiloyError,
    LangModel,
    add_agent_provider,
    add_tool_provider,
    register_a2a,
    register_mcp_stdio,
    register_mcp_streamable_http,
    register_tool,
    remove_lang_model_provider,
    set_lang_model_provider,
    unregister_mcp,
)

__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentRun",
    "AiloyError",
    "LangModel",
    "add_agent_provider",
    "add_tool_provider",
    "virtx",
    "register_a2a",
    "register_mcp_stdio",
    "register_mcp_streamable_http",
    "register_tool",
    "remove_lang_model_provider",
    "set_lang_model_provider",
    "types",
    "unregister_mcp",
]
