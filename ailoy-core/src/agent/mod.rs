//! What an agent is ([`AgentSpec`]) and what it carries from one turn to the next
//! ([`AgentState`]), with what a server takes and answers about each.

mod spec;
mod state;

pub use spec::*;
pub use state::*;
