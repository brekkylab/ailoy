//! An agent on [`model`](crate::model): a spec's model is a desc built through
//! [`create_lang_model`](crate::model::create_lang_model).
//!
//! ```ignore
//! use ailoy::agent::*;
//!
//! let spec = AgentSpec::new("anthropic/claude-sonnet-5-5").instruction("Be brief.").system_tools();
//! let mut agent = Agent::try_with_state(spec, AgentState::new().with_console(console)).await?;
//! let mut turn = agent.run(Message::new(Role::User).with_contents([Part::text("Hi")]));
//! while let Some(output) = turn.next().await { /* … */ }
//! ```

mod builder;
mod context;
mod rt;
mod skill;
mod spec;
mod state;

pub use builder::AgentBuilder;
pub use context::ContextManager;
pub use rt::*;
pub use spec::*;
pub use state::*;
