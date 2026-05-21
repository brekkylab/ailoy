use std::path::PathBuf;

use crate::{
    agent::{Agent, AgentProvider, AgentSpec, ContextManager},
    message::Message,
    runenv::{FileEntry, RunEnv},
    tool::{ToolDesc, WebSearchEngineKind},
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

    runenv: Option<RunEnv>,

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

    pub fn web_search_tool(mut self, engines: Vec<WebSearchEngineKind>) -> Self {
        self.spec = self.spec.web_search_tool(engines);
        self
    }

    pub fn web_fetch_tool(mut self) -> Self {
        self.spec = self.spec.web_fetch_tool();
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

    /// Use this [`RunEnv`] for tool execution instead of the default local runenv.
    /// `RunEnv` is cheaply cloneable, so the same underlying VM can be shared between
    /// multiple agents by cloning the value.
    pub fn runenv(mut self, runenv: RunEnv) -> Self {
        self.runenv = Some(runenv);
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

    /// Sampling temperature forwarded to the language model on every call.
    /// See [`AgentSpec::temperature`].
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.spec = self.spec.temperature(temperature);
        self
    }

    /// Top-p (nucleus) sampling parameter forwarded to the language model on
    /// every call. See [`AgentSpec::top_p`].
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.spec = self.spec.top_p(top_p);
        self
    }

    /// Top-k sampling parameter forwarded to the language model on every call.
    /// See [`AgentSpec::top_k`].
    pub fn top_k(mut self, top_k: u64) -> Self {
        self.spec = self.spec.top_k(top_k);
        self
    }

    pub fn response_format(mut self, fmt: crate::lang_model::ResponseFormat) -> Self {
        self.spec = self.spec.response_format(fmt);
        self
    }

    /// Append a single pre-fill [`FileEntry`] to `spec.files`.  The file is
    /// written into the runenv on the agent's first `run`.
    pub fn file(mut self, entry: FileEntry) -> Self {
        self.spec.files.push(entry);
        self
    }

    /// Append several [`FileEntry`]s to `spec.files`.
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
    use crate::{agent::AgentCard, lang_model::LangModelProvider, message::Role};

    const TEST_MODEL: &str = "openai/gpt-4o-mini";

    fn dummy_provider() -> AgentProvider {
        let mut provider = AgentProvider::new();
        provider
            .models
            .insert(TEST_MODEL.into(), LangModelProvider::openai("dummy".into()));
        provider
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
            .runenv(RunEnv::local())
            .build()
            .unwrap();
        // Smoke check: runenv is plugged in and usable.
        let handle = agent.state.runenv.get().await.expect("runenv boot failed");
        let result = handle
            .exec("sh".into(), vec!["-c".into(), "echo ok".into()], None)
            .await
            .expect("exec failed");
        assert!(result.stdout.contains("ok"));
    }

    /// Helper: build a SKILL.md body with the given name/description/body.
    fn skill_md(name: &str, desc: &str, body: &str) -> Vec<u8> {
        format!("---\nname: {name}\ndescription: {desc}\n---\n{body}").into_bytes()
    }

    /// `.skill(dir, [SKILL.md])` declares a skill and seeds its content in
    /// one call; the spec carries both pieces and the system instruction
    /// lists the skill.
    #[tokio::test]
    async fn test_file_skill_seeds_spec() {
        let dir = tempfile::tempdir().unwrap();
        let greet_dir = dir.path().join("skills/greet");
        let skill_path = greet_dir.join("SKILL.md");
        let agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(RunEnv::local())
            .skill(
                &greet_dir,
                [FileEntry::new(
                    &skill_path,
                    skill_md("greet", "Say hello.", "# greet\nbody\n"),
                )],
            )
            .build()
            .unwrap();

        assert_eq!(agent.spec().files.len(), 1);
        assert_eq!(agent.spec().files[0].path, skill_path);
        assert_eq!(agent.spec().skills, vec![greet_dir.clone()]);

        let sys = system_text(&agent).expect("expected system message");
        assert!(sys.contains("Available Skills"));
        assert!(sys.contains("greet"));
    }

    /// With no skills declared, the Available Skills block is omitted
    /// entirely (no noise in the system message).
    #[tokio::test]
    async fn test_no_skill_block_when_nothing_declared() {
        let agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .build()
            .unwrap();
        // No instruction, no skills → no system message at all.
        assert!(agent.get_history().is_empty());
    }

    /// Per-agent skill isolation: parent and sub each declare their own
    /// skill dir explicitly (no nesting convention).  Parent's system
    /// instruction lists only its own skill.
    #[tokio::test]
    async fn test_per_agent_skill_isolation() {
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
            .provider(dummy_provider())
            .runenv(RunEnv::local())
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

        // Parent's system message references only its own skill, not the sub's.
        let sys = system_text(&parent).expect("parent system message");
        assert!(sys.contains("a_skill"), "parent should advertise a_skill");
        assert!(
            !sys.contains("b_skill"),
            "parent must NOT see child's skill in its instruction: {sys}"
        );
    }

    /// Same-named skills (last path segment "foo") declared on parent and
    /// sub at explicit, disjoint paths carry different content without
    /// conflict — the new design relies on the user picking distinct paths,
    /// no automatic nesting.
    #[tokio::test]
    async fn test_same_skill_name_across_levels_no_conflict() {
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
            .provider(dummy_provider())
            .runenv(RunEnv::local())
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

        // Parent + sub specs hold distinct files at distinct paths.
        let parent_entry = parent
            .spec()
            .files
            .iter()
            .find(|f| f.path == parent_foo)
            .expect("parent file");
        let child_entry = parent.spec().subagents[0]
            .files
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

    /// Two agents built with the same cloned `RunEnv` see each other's
    /// filesystem writes — confirms `.runenv()` carries the VM reference
    /// through to the agent's `state.runenv`, and that cloning the v2
    /// `RunEnv` shares the underlying container.
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn test_builder_shared_arc_sandbox() {
        use crate::runenv::SandboxConfig;

        let runenv = RunEnv::sandbox(SandboxConfig::default())
            .await
            .expect("sandbox creation failed");

        let sub_agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(runenv.clone())
            .build()
            .unwrap();

        let parent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(runenv.clone())
            .build()
            .unwrap();

        // Write through the underlying VM, read back through parent's runenv.
        let handle = runenv.get().await.expect("runenv boot failed");
        handle
            .write(
                std::path::Path::new("/workspace/shared_test.txt"),
                b"shared_ok",
            )
            .await
            .expect("write failed");

        let bytes = parent
            .state
            .runenv
            .get()
            .await
            .expect("runenv boot failed")
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
            .get()
            .await
            .expect("runenv boot failed")
            .write(std::path::Path::new("/workspace/shared.txt"), b"shared_ok")
            .await
            .expect("write failed");

        let bytes = sub_agent
            .state
            .runenv
            .get()
            .await
            .expect("runenv boot failed")
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
