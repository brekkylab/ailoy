use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    agent::{Agent, AgentCard, AgentState},
    lang_model::LangModel,
    message::{Message, Part, Role},
    runenv::RunEnv,
    tool::Tool,
    tool_impl::make_subagent_tool,
};

/// Spec-free builder for [`Agent`].
///
/// Accepts runtime objects directly — [`LangModel`], [`Tool`], and optionally a
/// sandbox — without going through [`AgentSpec`] or [`AgentProvider`].  This is
/// the preferred construction path when the caller already holds materialised
/// runtime objects (e.g. from an existing session or an external DI container).
///
/// Use [`crate::agent::Agent::try_with_provider`] / [`Agent::try_new`] when you
/// want the spec/provider path (YAML-friendly, string-keyed).
pub struct AgentBuilder {
    model: LangModel,
    instruction: Option<String>,
    tools: Vec<Tool>,
    runenv: Option<Arc<dyn RunEnv>>,
}

impl AgentBuilder {
    /// Create a builder.  `model` is mandatory — every agent needs a language model.
    pub fn new(model: LangModel) -> Self {
        Self {
            model,
            instruction: None,
            tools: Vec::new(),
            runenv: None,
        }
    }

    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.instruction = Some(inst.into());
        self
    }

    /// Register a pre-built [`Tool`].  `ToolFactory` is not accepted here — if you
    /// need a factory-derived tool, call `factory.make(&spec)` first.
    pub fn tool(mut self, t: Tool) -> Self {
        self.tools.push(t);
        self
    }

    pub fn tools(mut self, ts: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.extend(ts);
        self
    }

    /// Register an already-built [`Agent`] as a subagent tool.
    ///
    /// If both parent and subagent should share a sandbox, pass the same
    /// `Arc<Sandbox>` to each builder via `.sandbox(arc.clone())`.
    pub fn subagent(mut self, card: AgentCard, agent: Agent) -> Self {
        let sub = Arc::new(Mutex::new(agent));
        self.tools.push(make_subagent_tool(card, sub));
        self
    }

    /// Set the sandbox for this agent.
    ///
    /// Pass an `Arc<Sandbox>` to share an existing VM, or a `SandboxConfig` to
    /// create a dedicated VM during [`build`](Self::build).
    pub fn runenv(mut self, runenv: Arc<dyn RunEnv>) -> Self {
        self.runenv = Some(runenv);
        self
    }

    /// Materialise the agent.  All runtime objects must already be present in the
    /// builder; no spec or provider lookup is performed.
    pub async fn build(self) -> anyhow::Result<Agent> {
        let history = self
            .instruction
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        #[allow(unused_mut)]
        let mut state = AgentState::new().history(history);

        Ok(Agent::from_parts(self.model, self.tools, state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang_model::{LangModel, LangModelProvider};

    fn test_model() -> LangModel {
        LangModel::new(
            "gpt-4o-mini".to_string(),
            LangModelProvider::openai("dummy".into()),
        )
    }

    #[tokio::test]
    async fn test_builder_no_sandbox() {
        let agent = AgentBuilder::new(test_model())
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
        let agent = AgentBuilder::new(test_model()).build().await.unwrap();
        assert!(agent.get_history().is_empty());
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn test_builder_shared_arc_sandbox() {
        use crate::runenv::{Sandbox, SandboxConfig};

        let sandbox = Arc::new(
            Sandbox::new(SandboxConfig {
                ..Default::default()
            })
            .await
            .expect("sandbox creation failed"),
        );

        let sub_agent = AgentBuilder::new(test_model())
            .runenv(sandbox.clone())
            .build()
            .await
            .unwrap();

        let parent = AgentBuilder::new(test_model())
            .runenv(sandbox.clone())
            .subagent(
                AgentCard {
                    name: "sub".into(),
                    description: "test subagent".into(),
                    skills: vec![],
                },
                sub_agent,
            )
            .build()
            .await
            .unwrap();

        // Verify shared sandbox: write through vm, read through parent's runenv.
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
    }

    /// Parent and subagent built with the same `Arc<Sandbox>` share the VM's
    /// filesystem: a file written through the parent's runenv is immediately
    /// visible when read through the subagent's runenv.
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn test_parent_and_subagent_share_sandbox_filesystem() {
        use std::sync::Arc;

        use crate::runenv::{Sandbox, SandboxConfig};

        let sandbox = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );

        // Build subagent first; clone the runenv Arc before handing it to the parent.
        let sub = AgentBuilder::new(test_model())
            .runenv(sandbox.clone())
            .build()
            .await
            .unwrap();
        let sub_runenv = sub.state.runenv.clone();

        let parent = AgentBuilder::new(test_model())
            .runenv(sandbox.clone())
            .subagent(
                AgentCard {
                    name: "sub".into(),
                    description: "test subagent".into(),
                    skills: vec![],
                },
                sub,
            )
            .build()
            .await
            .unwrap();

        // Write a file through the parent's runenv, read it back through the
        // subagent's runenv clone — both sides must observe the same filesystem state.
        parent
            .state
            .runenv
            .write(std::path::Path::new("/workspace/shared.txt"), b"shared_ok")
            .await
            .expect("write failed");

        let bytes = sub_runenv
            .read(std::path::Path::new("/workspace/shared.txt"))
            .await
            .expect("subagent runenv should see file written by parent");

        assert!(
            bytes.starts_with(b"shared_ok"),
            "subagent runenv did not see the file written by parent, got: {bytes:?}"
        );
    }
}
