use crate::{
    agent::{Agent, AgentSpec, AgentState, ContextManager},
    browser::{Browser, BrowserSession},
    datatype::Value,
    memory::Memory,
    message::Message,
    model::ThinkingEffort,
    tool::{ToolDesc, WebSearchEngineKind},
};

/// Fluent builder over [`AgentSpec`] for [`Agent`].
///
/// For assembling an agent inline; with a fully-formed [`AgentSpec`], call
/// [`Agent::try_new`] / [`Agent::try_with_state`] directly.
///
/// # Examples
///
/// ```rust,no_run
/// # use ailoy::agent::AgentBuilder;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// // The model is built through the provider set for its org (see `ailoy::model`), each
/// // tool through the provider set for its name (see `ailoy::tool`).
/// let agent = AgentBuilder::new("anthropic/claude-sonnet-5-5")
///     .instruction("Be brief.")
///     .web_fetch_tool()
///     .build()
///     .await?;
/// #   Ok(())
/// # }
/// ```
pub struct AgentBuilder {
    spec: AgentSpec,

    history: Vec<Message>,

    sandbox: Option<virtx::Sandbox>,

    memory: Option<Memory>,

    context_manager: Option<ContextManager>,

    browser: Option<Browser>,
}

impl AgentBuilder {
    /// Create a builder for the model `model` names (e.g. `"anthropic/claude-sonnet-5-5"`), as
    /// [`create_lang_model`](crate::model::create_lang_model) takes it.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            spec: AgentSpec::new(model),
            history: Vec::new(),
            sandbox: None,
            memory: None,
            context_manager: None,
            browser: None,
        }
    }

    /// Set the system instruction stored on the spec.
    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.spec = self.spec.instruction(inst);
        self
    }

    pub fn tool(mut self, desc: ToolDesc) -> Self {
        self.spec = self.spec.tool(desc);
        self
    }

    pub fn tools(mut self, descs: impl IntoIterator<Item = ToolDesc>) -> Self {
        self.spec = self.spec.tools(descs);
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

    /// Seed the agent's [`AgentState::history`] (e.g. for resuming a prior session).
    /// A leading system message here overrides the one the spec's instruction would
    /// produce; otherwise the instruction is still seeded, at the front of this history.
    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Run this agent's sandbox tools in `sandbox`, with whatever volumes it was made with.
    ///
    /// Nothing makes a sandbox implicitly, since that means choosing a virtx server.
    /// Without one, pure tools still run and sandbox tools fail with an error.
    pub fn sandbox(mut self, sandbox: virtx::Sandbox) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Give the agent a browser and the `browser_*` tools. The tools reach the
    /// browser from this process, not from the [`sandbox`](Self::sandbox). Nothing starts until
    /// the agent first uses it, or [`Agent::start_browser`].
    pub fn browser(mut self, browser: Browser) -> Self {
        self.browser = Some(browser);
        self
    }

    /// Let this agent remember into `memory`.
    ///
    /// Adds the `mem_search` and `mem_insert` tools; they are not spec tools because the
    /// store is a per-agent value, not a name with a [`ToolProvider`](crate::tool::ToolProvider).
    ///
    /// The store must already exist (`mem init`); a missing file is reported by the first
    /// memory tool call, not by [`build`](Self::build). Memory tools run on the
    /// [`sandbox`](Self::sandbox), so one is required too.
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

    /// How much the model thinks before it answers. See [`AgentSpec::thinking_effort`].
    pub fn thinking_effort(mut self, effort: ThinkingEffort) -> Self {
        self.spec = self.spec.thinking_effort(effort);
        self
    }

    /// JSON schema the model's answers must match. See [`AgentSpec::output_schema`].
    pub fn output_schema(mut self, schema: Value) -> Self {
        self.spec = self.spec.output_schema(schema);
        self
    }

    /// Materialise the agent via [`Agent::try_with_state`], which is async
    /// because it builds the model and reads the skills through the sandbox.
    pub async fn build(self) -> anyhow::Result<Agent> {
        let Self {
            spec,
            history,
            sandbox,
            memory,
            context_manager,
            browser,
        } = self;

        let mut state = AgentState::new().with_history(history);
        if let Some(sandbox) = sandbox {
            state = state.with_sandbox(sandbox);
        }
        if let Some(memory) = memory {
            state = state.with_memory(memory);
        }
        if let Some(browser) = browser {
            browser.check()?;
            state = state.with_browser(BrowserSession::new(browser));
        }

        let mut agent = Agent::try_with_state(spec, state).await?;
        agent.set_context_manager(context_manager);
        Ok(agent)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{future::BoxFuture, stream::BoxStream};

    use super::*;
    use crate::{
        message::{MessageDeltaOutput, MessageOutput, Part, Role},
        model::{InferLangModel, LangModelOptions, LangModelProvider, set_lang_model_provider},
        test_sandbox,
    };

    /// The org the tests build their model under.
    const TEST_ORG: &str = "agent-builder-tests";

    /// A model the builder tests build but never run.
    struct Unused;

    impl InferLangModel for Unused {
        fn infer(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
            unimplemented!()
        }

        fn infer_stream(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
            unimplemented!()
        }
    }

    fn builder() -> AgentBuilder {
        set_lang_model_provider(
            TEST_ORG,
            LangModelProvider::custom(|_, _| Ok(Arc::new(Unused))),
        );
        AgentBuilder::new(format!("{TEST_ORG}/any"))
    }

    fn system_text(agent: &Agent) -> Option<String> {
        let m = agent.history().first()?;
        if m.role != Role::System {
            return None;
        }
        m.contents
            .iter()
            .find_map(|p| p.as_text())
            .map(str::to_string)
    }

    fn msg(role: Role, text: &str) -> Message {
        Message::new(role).with_contents([Part::text(text)])
    }

    #[tokio::test]
    async fn test_simple_builder() {
        let agent = builder()
            .instruction("You are a test agent.")
            .build()
            .await
            .unwrap();

        let history = agent.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }

    #[tokio::test]
    async fn test_builder_no_instruction() {
        let agent = builder().build().await.unwrap();
        assert!(agent.history().is_empty());
    }

    /// `sandbox()` puts the supplied sandbox into `state.sandbox`.
    #[tokio::test]
    async fn test_builder_sandbox_is_applied() {
        let agent = builder()
            .sandbox(test_sandbox().await)
            .build()
            .await
            .unwrap();

        let sandbox = agent
            .state
            .sandbox
            .as_ref()
            .expect("the supplied sandbox is held");
        let result = crate::console::exec(sandbox, ["sh", "-c", "echo ok"], None)
            .await
            .expect("exec failed");
        assert_eq!(result.stdout, "ok\n");
    }

    #[tokio::test]
    async fn test_builder_context_manager_is_applied() {
        let cm = ContextManager {
            max_input_tokens: 10_000,
            preserve_recent_turns: 2,
            ..Default::default()
        };
        let agent = builder().context_manager(cm).build().await.unwrap();
        assert!(agent.context_manager().is_some());
    }

    #[tokio::test]
    async fn test_instruction_seeded_into_history_without_system_message() {
        let agent = builder()
            .instruction("You are a test agent.")
            .history([msg(Role::User, "hello"), msg(Role::Assistant, "hi")])
            .build()
            .await
            .unwrap();

        let history = agent.history();
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
        let agent = builder()
            .instruction("spec instruction")
            .history([
                msg(Role::System, "stored instruction"),
                msg(Role::User, "hello"),
            ])
            .build()
            .await
            .unwrap();

        assert_eq!(agent.history().len(), 2);
        assert_eq!(system_text(&agent).as_deref(), Some("stored instruction"));
    }

    /// `memory()` lands on the state, which decides whether the agent gets memory tools.
    #[tokio::test]
    async fn test_builder_memory_is_applied() {
        let agent = builder()
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
        let agent = builder()
            .memory(std::path::Path::new("/work/notes.sqlite"))
            .build()
            .await
            .unwrap();

        assert_eq!(agent.state.memory, Some(Memory::new("/work/notes.sqlite")));
    }
}
