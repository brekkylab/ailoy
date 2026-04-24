use std::sync::Arc;

use tokio::sync::Mutex;

#[cfg(feature = "sandbox")]
use crate::sandbox::{Sandbox, SandboxSource};
use crate::{
    agent::{Agent, AgentCard, AgentState},
    lang_model::LangModel,
    message::{Message, Part, Role},
    tool::Tool,
    tool_impl::make_subagent_tool,
};

/// Spec-free builder for [`Agent`].
///
/// Accepts runtime objects directly — [`LangModel`], [`Tool`], and optionally an
/// [`Arc<Sandbox>`] or [`SandboxConfig`] — without going through [`AgentSpec`] or
/// [`AgentProvider`].  This is the preferred construction path when the caller already
/// holds materialised runtime objects (e.g. from an existing session or an external DI
/// container).
///
/// Use [`crate::agent::Agent::try_with_provider`] / [`Agent::try_new`] when you want the
/// spec/provider path (YAML-friendly, string-keyed).
pub struct AgentBuilder {
    model: LangModel,
    instruction: Option<String>,
    tools: Vec<Tool>,
    #[cfg(feature = "sandbox")]
    sandbox: Option<SandboxSource>,
}

impl AgentBuilder {
    /// Create a builder.  `model` is mandatory — every agent needs a language model.
    pub fn new(model: LangModel) -> Self {
        Self {
            model,
            instruction: None,
            tools: Vec::new(),
            #[cfg(feature = "sandbox")]
            sandbox: None,
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
    /// `Arc<Sandbox>` to each builder via `.with_sandbox(arc.clone())`.
    pub fn subagent(mut self, card: AgentCard, agent: Agent) -> Self {
        let sub = Arc::new(Mutex::new(agent));
        self.tools.push(make_subagent_tool(card, sub));
        self
    }

    /// Set the sandbox for this agent.
    ///
    /// Pass an `Arc<Sandbox>` to share an existing VM, or a `SandboxConfig` to
    /// create a dedicated VM during [`build`](Self::build).
    #[cfg(feature = "sandbox")]
    pub fn sandbox(mut self, src: impl Into<SandboxSource>) -> Self {
        self.sandbox = Some(src.into());
        self
    }

    /// Materialise the agent.  All runtime objects must already be present in the
    /// builder; no spec or provider lookup is performed.
    pub async fn build(self) -> anyhow::Result<Agent> {
        let history = self
            .instruction
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        #[cfg(feature = "sandbox")]
        let state = {
            let resolved_sandbox: Option<Arc<Sandbox>> = match self.sandbox {
                Some(SandboxSource::Shared(arc)) => Some(arc),
                Some(SandboxSource::Fresh(cfg)) => Some(Arc::new(Sandbox::new(cfg).await?)),
                None => None,
            };
            let mut s = AgentState::with_history(history);
            s.sandbox = resolved_sandbox;
            s
        };
        #[cfg(not(feature = "sandbox"))]
        let state = AgentState::with_history(history);

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

        #[cfg(feature = "sandbox")]
        assert!(agent.state.sandbox.is_none());

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
        use crate::sandbox::{Sandbox, SandboxConfig};

        let vm = Arc::new(
            Sandbox::new(SandboxConfig {
                name: Some(format!("ab-sh-{}", &uuid::Uuid::new_v4().to_string()[..8])),
                ..Default::default()
            })
            .await
            .expect("sandbox creation failed"),
        );

        let sub_agent = AgentBuilder::new(test_model())
            .sandbox(vm.clone())
            .build()
            .await
            .unwrap();

        let parent = AgentBuilder::new(test_model())
            .sandbox(vm.clone())
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

        assert!(
            Arc::ptr_eq(parent.state.sandbox.as_ref().unwrap(), &vm),
            "parent sandbox must be the shared Arc"
        );

        // Cleanup — unwrap the Arc (builder test owns the only strong ref here)
        if let Some(sb) = parent.state.sandbox {
            if let Ok(owned) = Arc::try_unwrap(sb) {
                let _ = owned.shutdown().await;
            }
        }
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn test_builder_fresh_config_creates_distinct_sandbox() {
        use crate::sandbox::SandboxConfig;

        let cfg1 = SandboxConfig::default();
        let cfg2 = SandboxConfig::default();

        let agent1 = AgentBuilder::new(test_model())
            .sandbox(cfg1)
            .build()
            .await
            .unwrap();

        let agent2 = AgentBuilder::new(test_model())
            .sandbox(cfg2)
            .build()
            .await
            .unwrap();

        let sb1 = agent1.state.sandbox.as_ref().unwrap();
        let sb2 = agent2.state.sandbox.as_ref().unwrap();
        assert!(
            !Arc::ptr_eq(sb1, sb2),
            "distinct configs must produce distinct Arc<Sandbox> instances"
        );
    }

    /// Parent and subagent built with the same `Arc<Sandbox>` share the VM's
    /// filesystem: a file written through the parent's sandbox is immediately
    /// visible when read through the subagent's sandbox.
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn test_parent_and_subagent_share_sandbox_filesystem() {
        use std::sync::Arc;

        use crate::sandbox::{Sandbox, SandboxConfig};

        let vm = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );

        // Build subagent first; clone its sandbox Arc before handing it to the parent.
        let sub = AgentBuilder::new(test_model())
            .sandbox(vm.clone())
            .build()
            .await
            .unwrap();
        let sub_sandbox = sub
            .state
            .sandbox
            .clone()
            .expect("subagent must have sandbox");

        let parent = AgentBuilder::new(test_model())
            .sandbox(vm.clone())
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
        let parent_sandbox = parent
            .state
            .sandbox
            .as_ref()
            .expect("parent must have sandbox");

        // Both arcs must be the same object.
        assert!(
            Arc::ptr_eq(parent_sandbox, &vm),
            "parent sandbox != shared vm"
        );
        assert!(
            Arc::ptr_eq(&sub_sandbox, &vm),
            "subagent sandbox != shared vm"
        );

        // Write a file through the parent's sandbox, read it back through the
        // subagent's sandbox — both sides must observe the same filesystem state.
        parent_sandbox
            .write_file("/workspace/shared.txt", b"shared_ok")
            .await
            .expect("write_file failed");

        let content = sub_sandbox
            .read_file("/workspace/shared.txt")
            .await
            .expect("subagent sandbox should see file written by parent");

        assert!(
            content.contains("shared_ok"),
            "subagent sandbox did not see the file written by parent, got: {content:?}"
        );
    }
}
