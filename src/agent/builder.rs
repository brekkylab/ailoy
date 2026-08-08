use std::sync::Arc;

use cortex::console::Console;
use tokio::sync::Mutex;

use crate::{
    agent::{Agent, AgentSpec, AgentState, ContextManager},
    message::Message,
    tool::{ToolDesc, WebSearchEngineKind},
};

/// Fluent builder over [`AgentSpec`] for [`Agent`].
///
/// This is a convenience wrapper around the spec/provider construction path — useful
/// when you want to assemble an agent inline rather than constructing an [`AgentSpec`]
/// up front.
///
/// When you already hold a fully-formed [`AgentSpec`], call
/// [`Agent::try_with_provider_name`] / [`Agent::try_new`] / [`Agent::try_with_provider`]
/// directly instead.
///
/// # Examples
///
/// ```rust,no_run
/// # use ailoy::{
/// #     agent::AgentBuilder,
/// #     tool::ToolDescBuilder,
/// #     to_value,
/// # };
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// // Use the global `"default"` agent-provider bundle (env-driven lang-model
/// // registry + built-in tools).  Register additional named providers via
/// // `get_lm_providers_mut()` / `get_tool_providers_mut()` /
/// // `get_agent_providers_mut()` and reference them by name with
/// // [`AgentBuilder::agent_provider`].
/// let agent = AgentBuilder::new("openai/gpt-4o")
///     .tool(ToolDescBuilder::new("web_search")
///         .description("Search the web.")
///         .parameters(to_value!({ "type": "object", "properties": {} }))
///         .build()
///     )
///     .build()?;
/// #   Ok(())
/// # }
/// ```
pub struct AgentBuilder {
    spec: AgentSpec,

    /// Name of the [`AgentProvider`](crate::agent::AgentProvider) bundle to
    /// resolve at [`build`](Self::build) time.  Defaults to `"default"`.
    agent_provider: String,

    history: Vec<Message>,

    console: Option<Arc<Mutex<Option<Console>>>>,

    context_manager: Option<ContextManager>,
}

impl AgentBuilder {
    /// Create a builder for the given model identifier (e.g. `"openai/gpt-4o"`).
    /// The model must be resolvable by the [`AgentProvider`](crate::agent::AgentProvider)
    /// bundle selected at [`build`](Self::build) time.
    pub fn new(model: impl Into<String>) -> Self {
        let spec = AgentSpec::new(model);
        Self {
            spec,
            agent_provider: "default".to_string(),
            history: Vec::new(),
            console: None,
            context_manager: None,
        }
    }

    /// Select the [`AgentProvider`](crate::agent::AgentProvider) bundle to
    /// resolve against at [`build`](Self::build) time.  `name` must exist in
    /// the global registry exposed by
    /// [`get_agent_providers`](crate::agent::get_agent_providers); defaults
    /// to `"default"` if this method is never called.
    pub fn agent_provider(mut self, name: impl Into<String>) -> Self {
        self.agent_provider = name.into();
        self
    }

    /// Set the system instruction stored on the spec.
    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.spec = self.spec.instruction(inst);
        self
    }

    pub fn tool(mut self, desc: ToolDesc) -> Self {
        self.spec.tools.push(desc);
        self
    }

    pub fn tools(mut self, desc: impl IntoIterator<Item = ToolDesc>) -> Self {
        let mut desc = desc.into_iter().collect();
        self.spec.tools.append(&mut desc);
        self
    }

    /// Append the canonical local-execution toolset.
    /// See [`AgentSpec::system_tools`] for the per-family tool selection.
    pub fn system_tools(mut self) -> Self {
        self.spec = self.spec.system_tools();
        self
    }

    pub fn python_repl_tool(mut self) -> Self {
        self.spec = self.spec.python_repl_tool();
        self
    }

    pub fn shell_tool(mut self) -> Self {
        self.spec = self.spec.shell_tool();
        self
    }

    pub fn web_search_tool(mut self, engines: Vec<WebSearchEngineKind>) -> Self {
        self.spec = self.spec.web_search_tool(engines);
        self
    }

    pub fn web_fetch_tool(mut self) -> Self {
        self.spec = self.spec.web_fetch_tool();
        self
    }

    /// Append a sub-agent spec.  At [`build`](Self::build) time the sub-agent is
    /// materialised and registered as a callable tool, sharing the parent's machine.
    /// The sub-spec must carry an [`AgentCard`](crate::agent::AgentCard).
    pub fn subagent(mut self, spec: AgentSpec) -> Self {
        self.spec.subagents.push(spec);
        self
    }

    /// Seed the agent's [`AgentState::history`] (e.g. for resuming a prior session).
    /// A leading system message here overrides the one the spec's instruction would
    /// produce; otherwise the instruction is still seeded, at the front of this history.
    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Run this agent's console tools in `console`, which must already be started.
    ///
    /// Required for an agent whose tools need one — nothing builds a console on its
    /// own, because building one means choosing a console server to start, and that
    /// is the caller's decision. Without it, pure tools still run and a console tool
    /// fails saying so.
    pub fn console(mut self, console: Console) -> Self {
        self.console = Some(Arc::new(Mutex::new(Some(console))));
        self
    }

    /// Share a console slot with another `Agent` built elsewhere.
    pub fn shared_console(mut self, console: Arc<Mutex<Option<Console>>>) -> Self {
        self.console = Some(console);
        self
    }

    /// Set the context window management spec.
    pub fn context_manager(mut self, spec: ContextManager) -> Self {
        self.context_manager = Some(spec);
        self
    }

    /// Sampling temperature forwarded to the language model on every call.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.spec = self.spec.temperature(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.spec = self.spec.top_p(top_p);
        self
    }

    pub fn top_k(mut self, top_k: u64) -> Self {
        self.spec = self.spec.top_k(top_k);
        self
    }

    pub fn response_format(mut self, fmt: crate::lang_model::ResponseFormat) -> Self {
        self.spec = self.spec.response_format(fmt);
        self
    }

    /// Materialise the agent by dispatching to
    /// [`Agent::try_with_provider_name_and_state`] with a state assembled from
    /// the optional machine and history.
    pub fn build(self) -> anyhow::Result<Agent> {
        let Self {
            spec,
            agent_provider,
            history,
            console,
            context_manager,
        } = self;

        let mut state = AgentState::new();
        if let Some(c) = console {
            state = state.with_console_slot(c);
        }
        if !history.is_empty() {
            state = state.with_history(history);
        }

        let mut agent = Agent::try_with_provider_and_state(spec, &agent_provider, state)?;
        if context_manager.is_some() {
            agent.set_context_manager(context_manager);
        }
        Ok(agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::{AgentCard, AgentProvider, get_agent_providers_mut},
        lang_model::{LangModelProvider, get_lm_providers_mut},
        message::Role,
        test_console,
    };

    const TEST_MODEL: &str = "openai/gpt-4o-mini";
    const TEST_PROVIDER_NAME: &str = "agent_builder_tests";

    /// Register a dummy lang-model provider scoped to this test module and an
    /// `AgentProvider` bundle that points at it.  Each call is idempotent.
    fn ensure_dummy_provider() {
        let mut lmps = get_lm_providers_mut();
        if !lmps.contains_key(TEST_PROVIDER_NAME) {
            let mut lmp = LangModelProvider::new();
            lmp.insert(TEST_MODEL.into(), LangModelProvider::openai("dummy".into()));
            lmps.insert(TEST_PROVIDER_NAME.to_string(), lmp);
        }
        drop(lmps);
        let mut aps = get_agent_providers_mut();
        aps.entry(TEST_PROVIDER_NAME.to_string())
            .or_insert_with(|| AgentProvider::new(TEST_PROVIDER_NAME, "default"));
    }

    fn system_text(agent: &Agent) -> Option<String> {
        let history = agent.get_history();
        let m = history.first()?;
        if m.role != Role::System {
            return None;
        }
        m.contents
            .iter()
            .find_map(|p| p.as_text())
            .map(str::to_string)
    }

    #[tokio::test]
    async fn test_simple_builder() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .instruction("You are a test agent.")
            .build()
            .unwrap();

        let history = agent.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }

    #[tokio::test]
    async fn test_builder_no_instruction() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .build()
            .unwrap();
        assert!(agent.get_history().is_empty());
    }

    /// `console()` carries the supplied console through to `state.console`, already
    /// filled in — where an agent left to itself would build one on first run.
    #[tokio::test]
    async fn test_builder_console_is_applied() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .console(test_console().await)
            .build()
            .unwrap();

        let mut guard = agent.state.console.lock().await;
        let console = guard.as_mut().expect("the supplied console is in the slot");
        let result = console
            .exec(["sh", "-c", "echo ok"], None)
            .await
            .expect("exec failed");
        assert_eq!(result.stdout, b"ok\n");
    }

    #[tokio::test]
    async fn test_builder_subagent_in_spec() {
        use crate::agent::AgentSpec;

        ensure_dummy_provider();
        let sub_spec = AgentSpec::new(TEST_MODEL).card(AgentCard {
            name: "child".into(),
            description: "child agent".into(),
            skills: vec![],
        });

        AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .subagent(sub_spec)
            .build()
            .unwrap();
    }

    #[tokio::test]
    async fn test_builder_context_manager_is_applied() {
        ensure_dummy_provider();
        let cm = ContextManager {
            max_input_tokens: 10_000,
            preserve_recent_turns: 2,
        };
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .context_manager(cm)
            .build()
            .unwrap();
        assert!(agent.get_context_manager().is_some());
    }

    fn msg(role: Role, text: &str) -> Message {
        Message::new(role).with_contents([crate::message::Part::text(text)])
    }

    #[tokio::test]
    async fn test_instruction_seeded_into_history_without_system_message() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .instruction("You are a test agent.")
            .history([msg(Role::User, "hello"), msg(Role::Assistant, "hi")])
            .build()
            .unwrap();

        let history = agent.get_history();
        assert_eq!(history.len(), 3);
        assert_eq!(
            system_text(&agent).as_deref(),
            Some("You are a test agent.")
        );
        assert_eq!(history[1].role, Role::User);
        assert_eq!(history[2].role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_existing_system_message_is_not_replaced() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .instruction("spec instruction")
            .history([
                msg(Role::System, "stored instruction"),
                msg(Role::User, "hello"),
            ])
            .build()
            .unwrap();

        let history = agent.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(system_text(&agent).as_deref(), Some("stored instruction"));
    }
}
