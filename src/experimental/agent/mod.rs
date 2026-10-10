//! An agent on [`experimental::model`](super::model): a spec's model is a desc built through
//! [`create_lang_model`](super::model::create_lang_model).
//!
//! ```ignore
//! use ailoy::experimental::agent::*;
//!
//! let spec = AgentSpec::new("claude/sonnet").instruction("Be brief.").system_tools();
//! let mut agent = Agent::try_with_state(spec, AgentState::new().with_console(console)).await?;
//! let mut turn = agent.run(Message::new(Role::User).with_contents([Part::text("Hi")]));
//! while let Some(output) = turn.next().await { /* … */ }
//! ```

mod provider;
mod rt;
mod skill;
mod spec;
mod state;

pub use provider::*;
pub use rt::*;
pub use spec::*;
pub use state::*;
