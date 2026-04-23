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
//!   └─ ToolFactory ──make(&AgentSpec)──► Tool (runtime)
//!                                         └─ ToolFunc (behaviour)
//! ```

mod func;
mod provider;
mod rt;
mod toolset;

pub use func::*;
pub use provider::*;
pub use rt::*;
pub use toolset::*;
