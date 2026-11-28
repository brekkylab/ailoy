"""Agents that drive a language model through tool-augmented turns.

An ``Agent`` is made from a spec with ``Agent.from_spec`` and awaited; a turn is
``agent.run(query)``, iterated with ``async for``. A model is named ``<org>/<name>``
(``"anthropic/claude-sonnet-5-5"``) and runs on the backend set for its org with
``set_lang_model_provider``: by default its CLI, or its own API when it has none.
``LangModel.create(desc)`` builds one to run without an agent. A tool is given by name and made
through the provider of its name (``register_tool``, ``register_mcp_stdio``, ...); every
built-in tool is provided from the start.

Messages, specs and tool descriptions are dicts in their JSON form; ``ailoy.types`` spells
out their shapes.

Tools run commands in a sandbox from the ``virtx`` package, which ailoy depends on: each
agent's sandbox is made, as ``Agent.from_spec``'s ``sandbox`` says, with the ``virtx.Client``
given to ``set_virtx``, and goes with the agent.

See the Rust crate's documentation for details.
"""

from . import types
from ._ailoy import (
    Agent,
    AgentRun,
    AiloyError,
    LangModel,
    register_mcp_stdio,
    register_mcp_streamable_http,
    get_lang_model_provider,
    register_tool,
    remove_lang_model_provider,
    remove_tool_provider,
    set_lang_model_provider,
    set_virtx,
    set_web_search_provider,
    get_web_search_provider,
    unregister_mcp,
)

__all__ = [
    "Agent",
    "AgentRun",
    "AiloyError",
    "LangModel",
    "get_lang_model_provider",
    "register_mcp_stdio",
    "register_mcp_streamable_http",
    "register_tool",
    "remove_lang_model_provider",
    "remove_tool_provider",
    "set_lang_model_provider",
    "set_virtx",
    "set_web_search_provider",
    "get_web_search_provider",
    "types",
    "unregister_mcp",
]
