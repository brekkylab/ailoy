//! Tool abstractions and their relationships.
//!
//! A [`ToolProvider`] holds an ordered list of [`ToolProviderElem`] entries — each
//! describes a tool source (built-in, MCP server, A2A agent, or custom factory).
//! When an [`crate::agent::Agent`] is instantiated from an [`crate::agent::AgentSpec`],
//! each entry is resolved to a [`ToolFactory`] and then to a concrete [`Tool`] bound
//! to that spec.  The [`ToolFunc`] selected during that step decides the tool's
//! actual runtime behaviour.
//!
//! ```text
//! ToolProvider (ordered list of sources)
//!   └─ ToolProviderElem ──► ToolFactory ──► Tool (runtime)
//!                                            └─ ToolDesc (description)
//!                                            └─ ToolFunc (behaviour)
//! ```
//!
//! ## Lifecycle
//!
//! 1. **`ToolProvider` is created** — populated with [`ToolProviderElem`] entries
//!    via builder methods (`.bash()`, `.web_search()`, `.mcp_stdio(...)`,
//!    `.custom(factory)`, …) or by serde.
//! 2. **`Agent` is instantiated from an `AgentSpec`** — [`ToolProvider::make_runtime`]
//!    walks every element, asks it to build a [`ToolFactory`], and immediately calls
//!    [`ToolFactory::make`] with the spec, producing a concrete [`Tool`] for the agent.
//! 3. **`ToolFunc` drives execution** — the [`ToolFunc`] chosen during step 2
//!    determines the tool's actual runtime behaviour when [`Tool::call`] is invoked
//!    by the agent.

mod func;
mod provider;
mod rt;

pub use func::*;
pub use provider::*;
pub use rt::*;
