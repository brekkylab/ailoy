//! An agent on [`model`](crate::model): a spec's model is a desc built through
//! [`create_lang_model`](crate::model::create_lang_model), and each tool it names is made
//! through [`default_provider`].
//!
//! ```ignore
//! use ailoy_runner::agent::*;
//!
//! let spec = AgentSpec::new("anthropic/claude-sonnet-5-5").instruction("Be brief.").sandbox();
//! let mut agent = AgentRuntime::try_with_state(spec, AgentState::new(), Some(sandbox)).await?;
//! let mut turn = agent.run(Message::new(Role::User).with_contents([Part::text("Hi")]));
//! while let Some(output) = turn.next().await { /* … */ }
//! ```

mod image;
mod rt;
mod skill;
mod state;

use std::sync::{Arc, LazyLock, RwLock};

pub use ailoy_core::agent::{
    AgentListOutputItem, AgentProvider, AgentPutArgs, AgentSpec, AgentState, AppSessionCreateArgs,
    AppSessionCreateOutput, Refusal, SessionCreateArgs, SessionListOutputItem, SessionSandbox,
    TruncationPolicy,
};
pub use rt::*;

use crate::{
    model::Backend,
    tool::{Browser, Console, Web},
};

/// A provider of the console, web and browser tools, running each org's models through its CLI,
/// or its API when it has no CLI, and keeping its agents in memory: what [`default_provider`]
/// starts as, and what a server of its own starts from.
pub fn builtin_provider() -> AgentProvider {
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
    provider
}

/// The provider of this process, which an [`AgentRuntime`] makes the model and tools its spec
/// names through unless given another, and which a router may serve: the
/// [`builtin_provider`], as it is set from then on.
pub fn default_provider() -> &'static Arc<RwLock<AgentProvider>> {
    static PROVIDER: LazyLock<Arc<RwLock<AgentProvider>>> =
        LazyLock::new(|| Arc::new(RwLock::new(builtin_provider())));
    &PROVIDER
}
