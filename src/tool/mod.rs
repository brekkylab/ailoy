//! Tool abstractions and their relationships.
//!
//! A [`ToolSet`] is created upfront and holds [`ToolFactory`] instances by name.
//! When an [`crate::agent::Agent`] is instantiated from an [`crate::agent::AgentSpec`],
//! each [`ToolFactory`] in the set is asked to produce a concrete [`Tool`] for
//! that spec — binding the right [`ToolFunc`] to the agent's configuration
//! (for example, choosing a sandbox-aware variant when the spec declares a sandbox).
//! The [`ToolFunc`] is what ultimately decides the tool's runtime behaviour.
//!
//! ```text
//! ToolSet (holds factories by name)
//!   └─ ToolFactory ──► Tool (runtime)
//!                      └─ ToolDesc (description)
//!                      └─ ToolFunc (behaviour)
//! ```
//!
//! ## Lifecycle
//!
//! 1. **`ToolSet` is created** — populated with [`ToolFactory`] instances, each
//!    registered under its tool name.  No agent is involved yet.
//! 2. **`ToolFactory` lives in `ToolSet`** — it is a factory that knows *how* to
//!    build a [`Tool`], but defers construction until an agent's configuration
//!    is known.
//! 3. **`Agent` is instantiated from an `AgentSpec`** — at this point each
//!    [`ToolFactory`] in the set calls [`ToolFactory::make`] with the spec,
//!    producing a concrete [`Tool`] bound to that agent's configuration.
//! 4. **`ToolFunc` drives execution** — the [`ToolFunc`] selected during step 3
//!    determines the tool's actual runtime behaviour when [`Tool::call`] is
//!    invoked by the agent.

mod func;
mod provider;
mod rt;
mod toolset;

pub use func::*;
pub use provider::*;
pub use rt::*;
pub use toolset::*;
