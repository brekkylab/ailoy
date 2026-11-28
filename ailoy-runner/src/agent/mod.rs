//! An agent on [`model`](crate::model): a spec's model is a desc built through
//! [`create_lang_model`](crate::model::create_lang_model), and each tool it names is made
//! through [`default_provider`].
//!
//! ```ignore
//! use ailoy_runner::agent::*;
//!
//! let mut spec = AgentSpec::new("anthropic/claude-sonnet-5-5").instruction("Be brief.");
//! spec.use_sandbox = true;
//! let mut agent = Agent::try_with_state(spec, AgentState::new().with_sandbox(sandbox)).await?;
//! let mut turn = agent.run(Message::new(Role::User).with_contents([Part::text("Hi")]));
//! while let Some(output) = turn.next().await { /* … */ }
//! ```

mod builder;
mod image;
mod rt;
mod skill;
mod state;

use std::sync::{LazyLock, RwLock};

pub use ailoy_core::agent::{
    AgentListOutputItem, AgentProvider, AgentPutArgs, AgentSpec, Refusal, SessionCreateArgs,
    SessionListOutputItem, TruncationPolicy,
};
pub use builder::AgentBuilder;
pub use rt::*;
pub use state::*;

use crate::tool::{Browser, Console, Web};

/// The provider of this process, which an [`Agent`] makes the tools its spec names through. It
/// starts with the console, web and browser tools.
pub fn default_provider() -> &'static RwLock<AgentProvider> {
    static PROVIDER: LazyLock<RwLock<AgentProvider>> = LazyLock::new(|| {
        let mut provider = AgentProvider::new();
        provider.set_tool_provider(Console);
        provider.set_tool_provider(Web::default());
        provider.set_tool_provider(Browser);
        RwLock::new(provider)
    });
    &PROVIDER
}
