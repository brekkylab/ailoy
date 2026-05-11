use std::{path::PathBuf, sync::Arc};

use crate::{
    agent::{Agent, AgentProvider, AgentSpec, ContextManager, default_provider},
    message::Message,
    runenv::{FileEntry, Local, RunEnv},
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

    /// Override for the agent's root skill directory inside the runenv.
    /// When `None`, [`Agent::try_with_provider_and_runenv`] uses the default
    /// `/workspace/skills/` convention.
    skill_dir: Option<PathBuf>,
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
            skill_dir: None,
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

    /// Append a single pre-fill [`FileEntry`] to `spec.files`.  The file is
    /// written into the runenv at the agent's first run/snapshot.
    pub fn file(mut self, entry: FileEntry) -> Self {
        self.spec.files.push(entry);
        self
    }

    /// Append several [`FileEntry`]s to `spec.files`.
    pub fn files(mut self, entries: impl IntoIterator<Item = FileEntry>) -> Self {
        self.spec.files.extend(entries);
        self
    }

    /// Override the agent's root skill directory inside the runenv.  When
    /// unset, the default `/workspace/skills/` convention is used.  Useful
    /// for tests (point at a host tempdir) or for hosting multiple agents
    /// with disjoint skill layouts in a shared runenv.
    pub fn skill_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skill_dir = Some(dir.into());
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
            skill_dir,
        } = self;

        let mut agent = if let Some(skill_dir) = skill_dir {
            // Custom skill_dir requires the explicit internal constructor.
            // Materialise the provider/runenv defaults to feed it.
            let provider = match provider {
                Some(p) => p,
                None => default_provider().clone(),
            };
            let runenv = runenv.unwrap_or_else(|| Arc::new(Local {}));
            Agent::try_with_provider_and_runenv_and_skill_dir(spec, &provider, runenv, skill_dir)?
        } else {
            match (provider, runenv) {
                (None, None) => Agent::try_new(spec)?,
                (None, Some(runenv)) => Agent::try_with_runenv(spec, runenv)?,
                (Some(provider), None) => Agent::try_with_provider(spec, &provider)?,
                (Some(provider), Some(runenv)) => {
                    Agent::try_with_provider_and_runenv(spec, &provider, runenv)?
                }
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
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::{
        agent::AgentCard,
        lang_model::LangModelProvider,
        message::Role,
        runenv::Local,
    };

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
        m.contents.iter().find_map(|p| p.as_text()).map(str::to_string)
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

    /// Helper: build a SKILL.md body with the given name/description/body.
    fn skill_md(name: &str, desc: &str, body: &str) -> Vec<u8> {
        format!("---\nname: {name}\ndescription: {desc}\n---\n{body}").into_bytes()
    }

    /// `.file(...)` lands the entry on `spec.files`, and on first
    /// `snapshot()` (which fires the lazy materialise gate) the file
    /// appears in the runenv at the declared absolute path.
    #[tokio::test]
    async fn test_file_skill_seeds_spec_and_runenv() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills");
        let skill_path = skill_dir.join("greet/SKILL.md");
        let mut agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .skill_dir(&skill_dir)
            .file(FileEntry::new(
                &skill_path,
                skill_md("greet", "Say hello.", "# greet\nbody\n"),
            ))
            .build()
            .unwrap();

        // Spec carries the file (so it survives serialisation) — checked
        // before triggering any IO.
        assert_eq!(agent.spec().files.len(), 1);
        assert_eq!(agent.spec().files[0].path, skill_path);

        // The system instruction lists the skill (rendered at build, no IO).
        let sys = system_text(&agent).expect("expected system message");
        assert!(sys.contains("Available Skills"));
        assert!(sys.contains("greet"));

        // Snapshot triggers the lazy materialise gate; afterwards the file
        // exists on disk with frontmatter + body.
        let _ = agent.snapshot().await.unwrap();
        let bytes = tokio::fs::read(&skill_path).await.unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("---\nname: greet\ndescription: Say hello.\n---\n"));
        assert!(text.ends_with("# greet\nbody\n"));
    }

    /// Per-agent skill dirs: parent's skills at `<skill_dir>/<name>/SKILL.md`,
    /// sub-agent's at `<skill_dir>/<sub-card>/<name>/SKILL.md`.  Materialise
    /// walks the spec subtree, so sub-agent skill files exist even before
    /// the sub-agent is ever invoked.  Parent's system instruction lists
    /// only its own skills.
    #[tokio::test]
    async fn test_per_agent_skill_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills");
        let parent_skill_path = skill_dir.join("a_skill/SKILL.md");
        let sub_skill_path = skill_dir.join("child/b_skill/SKILL.md");

        let sub_spec = AgentSpec::new(TEST_MODEL)
            .card(AgentCard {
                name: "child".into(),
                description: "child agent".into(),
                skills: vec![],
            })
            .file(FileEntry::new(
                &sub_skill_path,
                skill_md("b_skill", "child only", "child body\n"),
            ));

        let mut parent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .skill_dir(&skill_dir)
            .file(FileEntry::new(
                &parent_skill_path,
                skill_md("a_skill", "parent only", "parent body\n"),
            ))
            .subagent(sub_spec)
            .build()
            .unwrap();

        // Trigger lazy materialise — walks parent's spec tree recursively.
        let _ = parent.snapshot().await.unwrap();

        // Both files exist, in their own dirs.
        assert!(
            tokio::fs::metadata(&parent_skill_path).await.is_ok(),
            "parent skill file should exist"
        );
        assert!(
            tokio::fs::metadata(&sub_skill_path).await.is_ok(),
            "sub skill file should exist at nested dir"
        );

        // Parent's system message references only its own skill, not the sub's.
        let sys = system_text(&parent).expect("parent system message");
        assert!(sys.contains("a_skill"), "parent should advertise a_skill");
        assert!(
            !sys.contains("b_skill"),
            "parent must NOT see child's skill in its instruction: {sys}"
        );
    }

    /// Same-named skills declared on parent and sub land in different dirs
    /// (parent + nested child) and can carry different content without
    /// conflict.
    #[tokio::test]
    async fn test_same_skill_name_across_levels_no_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills");
        let parent_foo = skill_dir.join("foo/SKILL.md");
        let child_foo = skill_dir.join("child/foo/SKILL.md");

        let sub_spec = AgentSpec::new(TEST_MODEL)
            .card(AgentCard {
                name: "child".into(),
                description: "child agent".into(),
                skills: vec![],
            })
            .file(FileEntry::new(
                &child_foo,
                skill_md("foo", "child foo", "CHILD\n"),
            ));

        let mut parent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .skill_dir(&skill_dir)
            .file(FileEntry::new(
                &parent_foo,
                skill_md("foo", "parent foo", "PARENT\n"),
            ))
            .subagent(sub_spec)
            .build()
            .unwrap();

        // Snapshot triggers recursive materialise.
        let _ = parent.snapshot().await.unwrap();

        let parent_text = tokio::fs::read_to_string(&parent_foo).await.unwrap();
        let child_text = tokio::fs::read_to_string(&child_foo).await.unwrap();
        assert!(parent_text.ends_with("PARENT\n"));
        assert!(child_text.ends_with("CHILD\n"));
        assert!(parent_text.contains("description: parent foo"));
        assert!(child_text.contains("description: child foo"));
    }

    /// Snapshot picks up a runtime-added skill: create a fresh skill directory
    /// with SKILL.md under the agent's skill_dir, then verify it shows up.
    #[tokio::test]
    async fn test_snapshot_picks_up_runtime_added_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills");
        let declared_path = skill_dir.join("declared/SKILL.md");
        let runtime_path = skill_dir.join("runtime/SKILL.md");

        let runenv: Arc<dyn RunEnv> = Arc::new(Local {});
        let mut agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .skill_dir(&skill_dir)
            .file(FileEntry::new(
                &declared_path,
                skill_md("declared", "stays", "BODY\n"),
            ))
            .build()
            .unwrap();

        // First snapshot triggers materialise (writes declared SKILL.md).
        let _ = agent.snapshot().await.unwrap();

        // Pretend the agent ran some bash and authored a brand-new skill.
        runenv.mkdir(&skill_dir.join("runtime")).await.unwrap();
        runenv
            .write(
                &runtime_path,
                &skill_md("runtime", "created at runtime", "live body\n"),
            )
            .await
            .unwrap();

        // Skill listing (always re-scans) sees both.
        let skills = agent.skills().await.unwrap();
        let names: Vec<_> = skills.iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"declared".to_string()));
        assert!(names.contains(&"runtime".to_string()));

        // Snapshot's file list also contains both SKILL.md files.
        let snap = agent.snapshot().await.unwrap();
        let paths: Vec<_> = snap.files.iter().map(|f| f.path.clone()).collect();
        assert!(paths.contains(&declared_path));
        assert!(paths.contains(&runtime_path));
    }

    /// Snapshot reflects a runtime overwrite: the file system wins over the
    /// originally declared content.  Write-once materialise ensures the
    /// second snapshot doesn't clobber the runtime overwrite.
    #[tokio::test]
    async fn test_snapshot_reflects_runtime_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills");
        let evolving_path = skill_dir.join("evolving/SKILL.md");

        let runenv: Arc<dyn RunEnv> = Arc::new(Local {});
        let mut agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .skill_dir(&skill_dir)
            .file(FileEntry::new(
                &evolving_path,
                skill_md("evolving", "v1", "ORIGINAL\n"),
            ))
            .build()
            .unwrap();

        // Bootstrap: first snapshot fires the materialise gate, writing v1.
        let _ = agent.snapshot().await.unwrap();

        // Overwrite the on-disk file with a v2 body.
        runenv
            .write(&evolving_path, &skill_md("evolving", "v2", "UPDATED\n"))
            .await
            .unwrap();

        // Second snapshot: gate already tripped, write-once preserves v2.
        let snap = agent.snapshot().await.unwrap();
        let evolving = snap
            .files
            .iter()
            .find(|f| f.path == evolving_path)
            .expect("evolving SKILL.md present in snapshot");
        let text = std::str::from_utf8(evolving.content.as_ref()).unwrap();
        assert!(text.contains("description: v2"));
        assert!(text.ends_with("UPDATED\n"));
    }

    /// Snapshot drops a skill the agent has deleted at runtime.
    #[tokio::test]
    async fn test_snapshot_drops_runtime_deleted_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills");
        let doomed_path = skill_dir.join("doomed/SKILL.md");

        let mut agent = AgentBuilder::new(TEST_MODEL)
            .provider(dummy_provider())
            .runenv(Local {})
            .skill_dir(&skill_dir)
            .file(FileEntry::new(
                &doomed_path,
                skill_md("doomed", "to be removed", "...\n"),
            ))
            .build()
            .unwrap();

        // Bootstrap: fire materialise to write the declared file.
        let _ = agent.snapshot().await.unwrap();

        // Delete the file (and its parent dir) on host.
        tokio::fs::remove_dir_all(skill_dir.join("doomed"))
            .await
            .unwrap();

        // Second snapshot: gate already tripped, no rewrite — the file is gone.
        let snap = agent.snapshot().await.unwrap();
        assert!(
            snap.files.iter().all(|f| f.path != doomed_path),
            "deleted skill should not be in snapshot: {:?}",
            snap.files.iter().map(|f| &f.path).collect::<Vec<_>>()
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
