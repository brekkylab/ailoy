//! An agent on [`model`](crate::model): a spec's model is a desc built through
//! [`create_lang_model`](crate::model::create_lang_model), and each tool it names is made
//! through [`default_provider`].
//!
//! ```ignore
//! use ailoy_runner::agent::*;
//!
//! let mut spec = AgentSpec::new("anthropic/claude-sonnet-5-5").instruction("Be brief.");
//! spec.use_sandbox = true;
//! let mut agent = AgentRuntime::try_with_state(spec, AgentState::new().with_sandbox(sandbox)).await?;
//! let mut turn = agent.run(Message::new(Role::User).with_contents([Part::text("Hi")]));
//! while let Some(output) = turn.next().await { /* … */ }
//! ```

mod image;
mod rt;
mod skill;
mod state;

use std::sync::{Arc, LazyLock, RwLock};

pub use ailoy_core::agent::{
    AgentListOutputItem, AgentProvider, AgentPutArgs, AgentSpec, Refusal, SessionCreateArgs,
    SessionListOutputItem, TruncationPolicy,
};
pub use rt::*;
pub use state::*;

use crate::{
    model::Backend,
    tool::{Browser, Console, Web},
};

/// The provider of this process, which an [`AgentRuntime`] makes the model and tools its spec
/// names through unless given another, and which a router may serve. It starts with the console,
/// web and browser tools, and runs each org's models through its CLI, or its API when it has no
/// CLI.
pub fn default_provider() -> &'static Arc<RwLock<AgentProvider>> {
    static PROVIDER: LazyLock<Arc<RwLock<AgentProvider>>> = LazyLock::new(|| {
        let mut provider = AgentProvider::new();
        provider.tools.set(Console);
        provider.tools.set(Web::default());
        provider.tools.set(Browser);
        for (org, backend) in [
            ("anthropic", Backend::Cli),
            ("openai", Backend::Cli),
            ("google", Backend::Cli),
            ("deepseek-ai", Backend::Api),
            ("moonshotai", Backend::Api),
            ("zai-org", Backend::Api),
            ("Qwen", Backend::Api),
            ("LGAI-EXAONE", Backend::Api),
        ] {
            provider.models.set(org, backend);
        }
        Arc::new(RwLock::new(provider))
    });
    &PROVIDER
}
