use std::path::PathBuf;

use crate::{
    agent::{Agent, AgentSpec, AgentState, ContextManager},
    message::Message,
    runenv::{FileEntry, Machine, SharedMachine},
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

    machine: Option<SharedMachine>,

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
            machine: None,
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
    /// When non-empty, this overrides the system message that the spec's instruction
    /// would otherwise produce.
    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Use this [`Machine`] for tool execution instead of a default [`Local`].
    /// Wraps the machine in `Arc<Mutex<>>` so sub-agents inherit the same VM.
    pub fn machine<M: Machine>(mut self, m: M) -> Self {
        self.machine = Some(SharedMachine::new(m));
        self
    }

    /// Use this pre-shared machine handle. Useful when the same VM should be
    /// shared with another `Agent` built elsewhere.
    pub fn shared_machine(mut self, m: SharedMachine) -> Self {
        self.machine = Some(m);
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

    pub fn file(mut self, entry: FileEntry) -> Self {
        self.spec.files.push(entry);
        self
    }

    pub fn files(mut self, entries: impl IntoIterator<Item = FileEntry>) -> Self {
        self.spec.files.extend(entries);
        self
    }

    /// Declare a skill at `dir` together with its pre-fill content.
    /// Writes through to [`AgentSpec::skill`].
    pub fn skill(
        mut self,
        dir: impl Into<PathBuf>,
        entries: impl IntoIterator<Item = FileEntry>,
    ) -> Self {
        self.spec = self.spec.skill(dir, entries);
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
            machine,
            context_manager,
        } = self;

        let mut state = AgentState::new();
        if let Some(m) = machine {
            state = state.with_runenv(m);
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
        runenv::Local,
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

    /// `machine()` carries the supplied [`Machine`] through to `state.machine`.
    #[tokio::test]
    async fn test_builder_machine_is_applied() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .machine(Local::new())
            .build()
            .unwrap();
        // Smoke check: machine is plugged in and usable.
        let mut guard = agent.state.machine.get().await;
        let console = guard.start().await.expect("machine start failed");
        let result = console
            .exec("sh".into(), vec!["-c".into(), "echo ok".into()], None)
            .await
            .expect("exec failed");
        assert!(result.stdout.contains("ok"));
    }

    fn skill_md(name: &str, desc: &str, body: &str) -> Vec<u8> {
        format!("---\nname: {name}\ndescription: {desc}\n---\n{body}").into_bytes()
    }

    #[tokio::test]
    async fn test_file_skill_seeds_spec() {
        ensure_dummy_provider();
        let dir = tempfile::tempdir().unwrap();
        let greet_dir = dir.path().join("skills/greet");
        let skill_path = greet_dir.join("SKILL.md");
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .machine(Local::new())
            .skill(
                &greet_dir,
                [FileEntry::new(
                    &skill_path,
                    skill_md("greet", "Say hello.", "# greet\nbody\n"),
                )],
            )
            .build()
            .unwrap();

        assert_eq!(agent.files().len(), 1);
        assert_eq!(agent.files()[0].path, skill_path);
        assert_eq!(agent.skills(), vec![greet_dir.clone()]);

        let sys = system_text(&agent).expect("expected system message");
        assert!(sys.contains("Available Skills"));
        assert!(sys.contains("greet"));
    }

    #[tokio::test]
    async fn test_no_skill_block_when_nothing_declared() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .build()
            .unwrap();
        assert!(agent.get_history().is_empty());
    }

    #[tokio::test]
    async fn test_per_agent_skill_isolation() {
        ensure_dummy_provider();
        let dir = tempfile::tempdir().unwrap();
        let parent_skill_dir = dir.path().join("parent_skills/a_skill");
        let parent_skill_path = parent_skill_dir.join("SKILL.md");
        let sub_skill_dir = dir.path().join("sub_skills/b_skill");
        let sub_skill_path = sub_skill_dir.join("SKILL.md");

        let sub_spec = AgentSpec::new(TEST_MODEL)
            .card(AgentCard {
                name: "child".into(),
                description: "child agent".into(),
                skills: vec![],
            })
            .skill(
                &sub_skill_dir,
                [FileEntry::new(
                    &sub_skill_path,
                    skill_md("b_skill", "child only", "child body\n"),
                )],
            );

        let parent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .machine(Local::new())
            .skill(
                &parent_skill_dir,
                [FileEntry::new(
                    &parent_skill_path,
                    skill_md("a_skill", "parent only", "parent body\n"),
                )],
            )
            .subagent(sub_spec)
            .build()
            .unwrap();

        let sys = system_text(&parent).expect("parent system message");
        assert!(sys.contains("a_skill"), "parent should advertise a_skill");
        assert!(
            !sys.contains("b_skill"),
            "parent must NOT see child's skill in its instruction: {sys}"
        );
    }

    #[tokio::test]
    async fn test_same_skill_name_across_levels_no_conflict() {
        ensure_dummy_provider();
        let dir = tempfile::tempdir().unwrap();
        let parent_foo_dir = dir.path().join("parent/foo");
        let child_foo_dir = dir.path().join("child/foo");
        let parent_foo = parent_foo_dir.join("SKILL.md");
        let child_foo = child_foo_dir.join("SKILL.md");

        let sub_spec = AgentSpec::new(TEST_MODEL)
            .card(AgentCard {
                name: "child".into(),
                description: "child agent".into(),
                skills: vec![],
            })
            .skill(
                &child_foo_dir,
                [FileEntry::new(
                    &child_foo,
                    skill_md("foo", "child foo", "CHILD\n"),
                )],
            );

        let parent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .machine(Local::new())
            .skill(
                &parent_foo_dir,
                [FileEntry::new(
                    &parent_foo,
                    skill_md("foo", "parent foo", "PARENT\n"),
                )],
            )
            .subagent(sub_spec)
            .build()
            .unwrap();

        let parent_entry = parent
            .files()
            .iter()
            .find(|f| f.path == parent_foo)
            .expect("parent file");
        let child_entry = parent
            .files()
            .iter()
            .find(|f| f.path == child_foo)
            .expect("child file");
        assert!(
            std::str::from_utf8(parent_entry.content.as_ref())
                .unwrap()
                .contains("description: parent foo")
        );
        assert!(
            std::str::from_utf8(child_entry.content.as_ref())
                .unwrap()
                .contains("description: child foo")
        );
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
}
