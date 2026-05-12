use std::sync::Arc;

use crate::{
    agent::{Agent, AgentProvider, AgentSpec, ContextManager},
    message::Message,
    runenv::RunEnv,
    tool::ToolDesc,
};

/// Fluent builder over [`AgentSpec`] for [`Agent`].
///
/// This is a convenience wrapper around the spec/provider construction path — useful
/// when you want to assemble an agent inline rather than constructing an [`AgentSpec`]
/// up front.
///
/// When you already hold a fully-formed [`AgentSpec`], call
/// [`Agent::try_with_provider`] / [`Agent::try_new`] directly instead.
///
/// # Examples
///
/// ```rust,no_run
/// # use ailoy::{
/// #     agent::{AgentBuilder, AgentProvider},
/// #     lang_model::LangModelProvider,
/// #     tool::ToolDescBuilder,
/// #     to_value,
/// # };
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let mut provider = AgentProvider::new();
/// provider.models.insert(
///     "openai/gpt-4o".into(),
///     LangModelProvider::openai(std::env::var("OPENAI_API_KEY")?),
/// );
///
/// let agent = AgentBuilder::new("openai/gpt-4o")
///     .provider(provider)
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

    provider: Option<AgentProvider>,

    history: Vec<Message>,

    runenv: Option<Arc<dyn RunEnv>>,

    context_manager: Option<ContextManager>,
}

impl AgentBuilder {
    /// Create a builder for the given model identifier (e.g. `"openai/gpt-4o"`).
    /// The model must be registered in the [`AgentProvider`] used at [`build`](Self::build) time.
    pub fn new(model: impl Into<String>) -> Self {
        let spec = AgentSpec::new(model);
        Self {
            spec,
            provider: None,
            history: Vec::new(),
            runenv: None,
            context_manager: None,
        }
    }

    /// Use this [`AgentProvider`] instead of the global [`default_provider`](crate::agent::default_provider).
    pub fn provider(mut self, provider: AgentProvider) -> Self {
        self.provider = Some(provider);
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

    pub fn web_search_tool(mut self) -> Self {
        self.spec = self.spec.web_search_tool();
        self
    }

    /// Append a sub-agent spec.  At [`build`](Self::build) time the sub-agent is
    /// materialised and registered as a callable tool, sharing the parent's [`RunEnv`].
    /// The sub-spec must carry an [`AgentCard`](crate::agent::AgentCard).
    pub fn subagent(mut self, spec: AgentSpec) -> Self {
        self.spec.subagents.push(spec);
        self
    }

    /// Seed the agent's [`AgentState::history`] (e.g. for resuming a prior session).
    /// When non-empty, this overrides the system message that the spec's instruction
    /// would otherwise produce.
    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Use this [`RunEnv`] for tool execution instead of the default [`Local`](crate::runenv::Local).
    /// Taken by value; wrap an existing [`Arc<Sandbox>`](crate::runenv::Sandbox) in a
    /// `Clone + RunEnv` newtype if you need to share the same VM across agents.
    pub fn runenv(mut self, runenv: impl RunEnv) -> Self {
        self.runenv = Some(Arc::new(runenv));
        self
    }

    /// Set the context window management spec.
    ///
    /// When set, the agent will automatically truncate history when the input token count
    /// exceeds `spec.max_input_tokens`, preserving the most recent turns.
    pub fn context_manager(mut self, spec: ContextManager) -> Self {
        self.context_manager = Some(spec);
        self
    }

    /// Materialise the agent by dispatching to the appropriate `Agent::try_*` constructor
    /// based on which optional fields were supplied.
    pub fn build(self) -> anyhow::Result<Agent> {
        let Self {
            spec,
            provider,
            history,
            runenv,
            context_manager,
        } = self;
        let mut agent = match (provider, runenv) {
            (None, None) => Agent::try_new(spec)?,
            (None, Some(runenv)) => Agent::try_with_runenv(spec, runenv)?,
            (Some(provider), None) => Agent::try_with_provider(spec, &provider)?,
            (Some(provider), Some(runenv)) => {
                Agent::try_with_provider_and_runenv(spec, &provider, runenv)?
            }
        };
        // Only override the spec-derived history (which seeds the system instruction)
        // when the caller explicitly supplied one — e.g. for session resumption.
        if !history.is_empty() {
            agent.state.history = history;
        }
        if context_manager.is_some() {
            agent.set_context_manager(context_manager);
        }
        Ok(agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{agent::AgentCard, lang_model::LangModelProvider, message::Role, runenv::Local};

    const TEST_MODEL: &str = "openai/gpt-4o-mini";

    fn dummy_provider() -> AgentProvider {
        let mut provider = AgentProvider::new();
        provider
            .models
            .insert(TEST_MODEL.into(), LangModelProvider::openai("dummy".into()));
        provider
    }

    #[tokio::test]
    async fn test_simple_builder() {
        let agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .instruction("You are a test agent.")
            .build()
            .unwrap();

        let history = agent.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }

    #[tokio::test]
    async fn test_builder_no_instruction() {
        let agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .build()
            .unwrap();
        assert!(agent.get_history().is_empty());
    }

    /// `runenv()` carries the supplied `RunEnv` through to `state.runenv`.
    #[tokio::test]
    async fn test_builder_runenv_is_applied() {
        let agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .build()
            .unwrap();
        // Smoke check: runenv is plugged in and usable.
        let result = agent
            .state
            .runenv
            .exec("sh".into(), vec!["-c".into(), "echo ok".into()], None)
            .await
            .expect("exec failed");
        assert!(result.stdout.contains("ok"));
    }

    /// Subagents declared in the spec are accepted by `build()` (full delegation
    /// is exercised by integration tests in [`crate::agent::rt`]).
    #[tokio::test]
    async fn test_builder_subagent_in_spec() {
        use crate::agent::AgentSpec;

        let sub_spec = AgentSpec::new(TEST_MODEL).card(AgentCard {
            name: "child".into(),
            description: "child agent".into(),
            skills: vec![],
        });

        AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .subagent(sub_spec)
            .build()
            .unwrap();
    }

    /// Newtype over `Arc<Sandbox>` so the same underlying VM can be handed to
    /// multiple builders via `.runenv()`.  `AgentBuilder::runenv` takes
    /// `impl RunEnv + 'static` by value, so a bare `Arc<Sandbox>` would need
    /// `RunEnv` to be implemented for `Arc<T>`; this wrapper sidesteps that.
    #[cfg(feature = "sandbox")]
    #[derive(Clone)]
    struct SharedSandbox(Arc<crate::runenv::Sandbox>);

    #[cfg(feature = "sandbox")]
    #[async_trait::async_trait]
    impl RunEnv for SharedSandbox {
        async fn exec(
            &self,
            program: String,
            args: Vec<String>,
            timeout: Option<u64>,
        ) -> anyhow::Result<crate::runenv::ExecResult> {
            self.0.exec(program, args, timeout).await
        }

        async fn ls(&self, path: &std::path::Path) -> anyhow::Result<Vec<crate::runenv::Dirent>> {
            self.0.ls(path).await
        }

        async fn mkdir(&self, path: &std::path::Path) -> anyhow::Result<()> {
            self.0.mkdir(path).await
        }

        async fn rmdir(&self, path: &std::path::Path) -> anyhow::Result<()> {
            self.0.rmdir(path).await
        }

        async fn read(&self, path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
            self.0.read(path).await
        }

        async fn write(&self, path: &std::path::Path, content: &[u8]) -> anyhow::Result<()> {
            self.0.write(path, content).await
        }
    }

    /// Two agents built with the same `Arc<Sandbox>` (via `SharedSandbox`) see
    /// each other's filesystem writes — confirms `.runenv()` carries the VM
    /// reference through to the agent's `state.runenv`.
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn test_builder_shared_arc_sandbox() {
        use crate::runenv::{Sandbox, SandboxConfig};

        let sandbox = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );
        let shared = SharedSandbox(sandbox.clone());

        let sub_agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(shared.clone())
            .build()
            .unwrap();

        let parent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(shared.clone())
            .build()
            .unwrap();

        // Write through the underlying VM, read back through parent's runenv.
        sandbox
            .write(
                std::path::Path::new("/workspace/shared_test.txt"),
                b"shared_ok",
            )
            .await
            .expect("write failed");

        let bytes = parent
            .state
            .runenv
            .read(std::path::Path::new("/workspace/shared_test.txt"))
            .await
            .expect("read failed");

        assert_eq!(
            bytes, b"shared_ok",
            "parent runenv must see file written through shared vm"
        );

        // And the subagent's runenv sees writes from the parent.
        parent
            .state
            .runenv
            .write(std::path::Path::new("/workspace/shared.txt"), b"shared_ok")
            .await
            .expect("write failed");

        let bytes = sub_agent
            .state
            .runenv
            .read(std::path::Path::new("/workspace/shared.txt"))
            .await
            .expect("subagent runenv should see file written by parent");

        assert!(
            bytes.starts_with(b"shared_ok"),
            "subagent runenv did not see the file written by parent, got: {bytes:?}"
        );
    }

    #[tokio::test]
    async fn test_builder_context_manager_is_applied() {
        let cm = ContextManager {
            max_input_tokens: 10_000,
            preserve_recent_turns: 2,
        };
        let agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .context_manager(cm)
            .build()
            .unwrap();
        assert!(agent.get_context_manager().is_some());
    }
}
