//! What an agent is ([`AgentSpec`]) and what it carries from one turn to the next
//! ([`AgentState`]), with what a server takes and answers about each, and what holds the tools a
//! spec names and the agents kept ([`AgentProvider`]).

mod provider;
mod spec;
mod state;

pub use provider::*;
pub use spec::*;
pub use state::*;
