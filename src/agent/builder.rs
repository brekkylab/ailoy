use std::sync::Arc;

use cortex::console::ConsoleClient;
use tokio::sync::Mutex;

use crate::{
    agent::{Agent, AgentSpec, AgentState, ContextManager},
    memory::Memory,
    message::Message,
    tool::{ToolDesc, WebSearchEngineKind},
};

/// Fluent builder over [`AgentSpec`] for [`Agent`].
///
/// For assembling an agent inline; with a fully-formed [`AgentSpec`], call
/// [`Agent::try_new`] / [`Agent::try_with_provider`] directly.
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
/// // Uses the `"default"` agent-provider bundle (env-driven lang models + built-in
/// // tools). Register others via `get_lm_providers_mut()` / `get_tool_providers_mut()` /
/// // `get_agent_providers_mut()` and select them with [`AgentBuilder::agent_provider`].
/// let agent = AgentBuilder::new("openai/gpt-4o")
///     .tool(ToolDescBuilder::new("web_search")
///         .description("Search the web.")
///         .parameters(to_value!({ "type": "object", "properties": {} }))
///         .build()
///     )
///     .build()
///     .await?;
/// #   Ok(())
/// # }
/// ```
pub struct AgentBuilder {
    spec: AgentSpec,

    /// [`AgentProvider`](crate::agent::AgentProvider) name resolved at [`build`](Self::build).
    agent_provider: String,

    history: Vec<Message>,

    console: Option<Arc<Mutex<Option<ConsoleClient>>>>,

    memory: Option<Memory>,

    context_manager: Option<ContextManager>,
}

impl AgentBuilder {
    /// Create a builder for `model` (e.g. `"openai/gpt-4o"`), which the selected
    /// [`AgentProvider`](crate::agent::AgentProvider) must resolve at [`build`](Self::build) time.
    pub fn new(model: impl Into<String>) -> Self {
        let spec = AgentSpec::new(model);
        Self {
            spec,
            agent_provider: "default".to_string(),
            history: Vec::new(),
            console: None,
            memory: None,
            context_manager: None,
        }
    }

    /// Select the [`AgentProvider`](crate::agent::AgentProvider) bundle (default
    /// `"default"`). `name` must be registered in
    /// [`get_agent_providers`](crate::agent::get_agent_providers) by [`build`](Self::build) time.
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

    /// Append a sub-agent spec, registered as a callable tool at [`build`](Self::build)
    /// time and sharing the parent's console. It must carry an
    /// [`AgentCard`](crate::agent::AgentCard).
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

    /// Run this agent's console tools in `console`. It need not be started: the first
    /// command that needs a booted session boots it.
    ///
    /// Nothing builds a console implicitly, since that means choosing a console server.
    /// Without one, pure tools still run and console tools fail with an error.
    pub fn console(mut self, console: ConsoleClient) -> Self {
        self.console = Some(Arc::new(Mutex::new(Some(console))));
        self
    }

    /// Share a console slot with another `Agent` built elsewhere.
    pub fn shared_console(mut self, console: Arc<Mutex<Option<ConsoleClient>>>) -> Self {
        self.console = Some(console);
        self
    }

    /// Let this agent remember into `memory`.
    ///
    /// Adds the `mem_search` and `mem_insert` tools; they are not spec tools because the
    /// store is a per-agent value, not a name in the [`ToolProvider`](crate::tool::ToolProvider).
    ///
    /// The store must already exist (`mem init`); a missing file is reported by the first
    /// memory tool call, not by [`build`](Self::build). Memory tools run on the
    /// [`console`](Self::console), so one is required too.
    pub fn memory(mut self, memory: impl Into<Memory>) -> Self {
        self.memory = Some(memory.into());
        self
    }

    /// Give this agent the skill in `dir`. See [`AgentSpec::skill`].
    pub fn skill(mut self, dir: impl Into<String>) -> Self {
        self.spec = self.spec.skill(dir);
        self
    }

    /// Set the context window management spec.
    pub fn context_manager(mut self, spec: ContextManager) -> Self {
        self.context_manager = Some(spec);
        self
    }

    /// The most tokens one reply may have, forwarded to the language model on every call.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.spec = self.spec.max_tokens(max_tokens);
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

    /// Turn on the model's thinking at `effort`, forwarded to the language model on every call.
    pub fn reasoning(mut self, effort: crate::lang_model::ReasoningEffort) -> Self {
        self.spec = self.spec.reasoning(effort);
        self
    }

    /// Materialise the agent via [`Agent::try_with_provider_and_state`], which is async
    /// because it reads the skills through the console.
    pub async fn build(self) -> anyhow::Result<Agent> {
        let Self {
            spec,
            agent_provider,
            history,
            console,
            memory,
            context_manager,
        } = self;

        let mut state = AgentState::new();
        if let Some(c) = console {
            state = state.with_console_slot(c);
        }
        if let Some(m) = memory {
            state = state.with_memory(m);
        }
        if !history.is_empty() {
            state = state.with_history(history);
        }

        let mut agent = Agent::try_with_provider_and_state(spec, &agent_provider, state).await?;
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

    /// Register a dummy lang-model provider and an `AgentProvider` pointing at it. Idempotent.
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
            .await
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
            .await
            .unwrap();
        assert!(agent.get_history().is_empty());
    }

    /// `console()` puts the supplied console into `state.console`.
    #[tokio::test]
    async fn test_builder_console_is_applied() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .console(test_console().await)
            .build()
            .await
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
            .await
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
            .await
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
            .await
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
            .await
            .unwrap();

        let history = agent.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(system_text(&agent).as_deref(), Some("stored instruction"));
    }

    /// `memory()` lands on the state, which decides whether the agent gets memory tools.
    #[tokio::test]
    async fn test_builder_memory_is_applied() {
        use crate::memory::Memory;

        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .memory(Memory::new("/work/notes.sqlite"))
            .build()
            .await
            .unwrap();

        assert_eq!(
            agent.state.memory,
            Some(Memory::new("/work/notes.sqlite")),
            "the memory the builder was given is the one on the state"
        );
    }

    /// A path converts into the equivalent `Memory`.
    #[tokio::test]
    async fn test_builder_memory_takes_a_path() {
        use crate::memory::Memory;

        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .memory(std::path::Path::new("/work/notes.sqlite"))
            .build()
            .await
            .unwrap();

        assert_eq!(agent.state.memory, Some(Memory::new("/work/notes.sqlite")));
    }
}
