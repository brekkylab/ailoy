//! A tool as described to a model ([`ToolDesc`]), what describes and makes it where the agent
//! runs ([`ToolProvider`]) and the tool as made, which a call runs ([`ToolFunc`]). A spec
//! names a tool and holds none of them.

mod desc;
mod func;
mod provider;

pub use desc::*;
pub use func::*;
pub use provider::*;
/// The sandbox a [`ToolFunc`] made with [`ToolFunc::new_with_sandbox`] runs in.
pub use virtx::Sandbox;
