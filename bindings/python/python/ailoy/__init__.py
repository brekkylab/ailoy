"""Agents that drive a language model through tool-augmented turns.

An ``Agent`` is built with an ``AgentBuilder`` and awaited; a turn is ``agent.run(query)``,
iterated with ``async for``. The model and the tools are resolved by name from process-wide
registries — ``register_lang_model``, ``register_tool``, ``register_mcp_stdio`` and the rest
— each of which starts with a ``"default"`` entry: the models whose API keys are in the
environment, and every built-in tool.

Messages, specs and tool descriptions are the dicts their JSON form is; ``ailoy.types``
spells out their shapes.

Where the tools run commands is a cortex console, which is in ``ailoy.cortex`` — built into
this package, so that an ``Agent`` can share its session.

The names and their behaviour are ailoy's own; see the Rust crate's documentation for the
long form.
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
