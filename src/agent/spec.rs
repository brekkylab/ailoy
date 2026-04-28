use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::agent::AgentCard;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub enum RunenvSpec {
    Local,

    Sandbox { key: Option<String> },
}

/// Defines the logical identity of an agent as configured by the user.
///
/// `AgentSpec` captures what makes an agent distinct — the language model it uses,
/// the system instruction that shapes its behavior, the set of tools it has access to,
/// and the sub-agents it can delegate work to.
/// Changing any of these fields changes the fundamental nature of the agent.
///
/// # `instruction` vs `card`
///
/// [`instruction`](AgentSpec::instruction) is *internal*: private guidance fed to the
/// model that callers never see.  It controls how this agent thinks and behaves.
///
/// [`card`](AgentSpec::card) is *external*: a public self-introduction that a calling
/// agent or orchestrator reads to decide whether to delegate work here.  Only sub-agents
/// need a card — a top-level agent has no caller to introduce itself to.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpec {
    /// Identifier of the language model (e.g. `"claude-sonnet-4-6"`)
    pub model: String,

    /// System prompt that shapes how the model works.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    /// Names of tools available to the agent
    pub tools: Vec<String>,

    /// Sub-agents available to the agent (each registered as a callable tool)
    pub subagents: Vec<AgentSpec>,

    /// Public self-introduction exposed to a calling agent or orchestrator.
    ///
    /// Only relevant when this agent acts as a sub-agent.
    /// `None` for top-level agents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<AgentCard>,

    /// Run environment.
    /// If not specified, the tools will be run in local machine.
    /// If specified, it'll use the run environment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runenv: Option<RunenvSpec>,
}

impl AgentSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instruction: None,
            tools: Vec::new(),
            subagents: Vec::new(),
            card: None,
            runenv: None,
        }
    }

    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.instruction = Some(inst.into());
        self
    }

    pub fn tool(mut self, tool: impl Into<String>) -> Self {
        self.tools.push(tool.into());
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = tools.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn subagent(mut self, spec: AgentSpec) -> Self {
        self.subagents.push(spec);
        self
    }

    pub fn subagents(mut self, specs: impl IntoIterator<Item = AgentSpec>) -> Self {
        self.subagents = specs.into_iter().collect();
        self
    }

    pub fn card(mut self, card: AgentCard) -> Self {
        self.card = Some(card);
        self
    }

    pub fn sandbox(mut self, sandbox: impl Into<String>) -> Self {
        self.runenv = Some(RunenvSpec::Sandbox {
            key: Some(sandbox.into()),
        });
        self
    }
}
