"""Agents that drive a language model through tool-augmented turns.

An ``Agent`` is built with an awaited ``AgentBuilder``; a turn is ``agent.run(query)``,
iterated with ``async for``. Models and tools resolve by name from process-wide registries
(``register_lang_model``, ``register_tool``, ``register_mcp_stdio``, ...), each starting with
a ``"default"`` entry: the models whose API keys are in the environment, and every built-in
tool.

Messages, specs and tool descriptions are dicts in their JSON form; ``ailoy.types`` spells
out their shapes.

Tools run commands in a cortex console from ``ailoy.cortex``, built into this package so an
``Agent`` can share its session.

See the Rust crate's documentation for details.
"""

from . import cortex, types
from ._ailoy import (
    Agent,
    AgentBuilder,
    AgentRun,
    AiloyError,
    add_agent_provider,
    add_lang_model_provider,
    add_tool_provider,
    register_a2a,
    register_lang_model,
    register_mcp_stdio,
    register_mcp_streamable_http,
    register_tool,
    unregister_mcp,
)

__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentRun",
    "AiloyError",
    "add_agent_provider",
    "add_lang_model_provider",
    "add_tool_provider",
    "cortex",
    "register_a2a",
    "register_lang_model",
    "register_mcp_stdio",
    "register_mcp_streamable_http",
    "register_tool",
    "types",
    "unregister_mcp",
]
