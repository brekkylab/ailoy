//! What an agent is ([`AgentSpec`]) and what it carries from one turn to the next
//! ([`AgentState`]), with what a server takes and answers about each, what holds the tools a spec
//! names ([`AgentProvider`]), and where the agents and their sessions are kept ([`AgentStore`]).

mod provider;
mod spec;
mod state;
mod store;

pub use provider::*;
pub use spec::*;
pub use state::*;
pub use store::*;
