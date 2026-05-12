use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentCard,
    tool::{
        ToolDesc, get_apply_patch_tool_desc, get_bash_tool_desc, get_edit_tool_desc,
        get_glob_tool_desc, get_grep_tool_desc, get_python_repl_tool_desc, get_read_tool_desc,
        get_web_search_tool_desc, get_write_tool_desc,
    },
};

/// Defines the logical identity of an agent as configured by the user.
///
/// `AgentSpec` captures what makes an agent distinct — the language model it uses,
/// the system instruction that shapes its behaviour, the set of tools it has access
/// to, and the sub-agents it can delegate work to.  Changing any of these fields
/// changes the fundamental nature of the agent.
///
/// Runtime concerns — credentials, tool sources, and the [`RunEnv`](crate::runenv::RunEnv)
/// — live on [`AgentProvider`](crate::agent::AgentProvider) and the constructors in
/// [`Agent`](crate::agent::Agent), not here.
///
/// # `instruction` vs `card`
///
/// [`instruction`](AgentSpec::instruction) is *internal*: private guidance fed to the
/// model that callers never see.  It controls how this agent thinks and behaves.
///
/// [`card`](AgentSpec::card) is *external*: a public self-introduction that a calling
/// agent or orchestrator reads to decide whether to delegate work here.  Sub-agents
/// must have a card — it supplies the name and description of the tool the parent
/// will call.  Top-level agents typically don't need one.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpec {
    /// Identifier of the language model (e.g. `"anthropic/claude-sonnet-4-6"`)
    pub model: String,

    /// System prompt that shapes how the model works.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    /// Tool descriptions exposed to the model. Each [`ToolDesc::name`] must match
    /// an entry registered in the [`AgentProvider`](crate::agent::AgentProvider)'s
    /// [`ToolProvider`](crate::tool::ToolProvider).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDesc>,

    /// Sub-agents available to the agent (each registered as a callable tool)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub subagents: Vec<AgentSpec>,

    /// Public self-introduction exposed to a calling agent or orchestrator.
    ///
    /// Only relevant when this agent acts as a sub-agent.
    /// `None` for top-level agents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<AgentCard>,
}

impl AgentSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instruction: None,
            tools: Vec::new(),
            subagents: Vec::new(),
            card: None,
        }
    }

    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.instruction = Some(inst.into());
        self
    }

    pub fn tool(mut self, tool: ToolDesc) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDesc>) -> Self {
        self.tools.append(&mut tools.into_iter().collect());
        self
    }

    /// Append the canonical local-execution toolset for the spec's model family.
    ///
    /// * `openai/*`: `bash`, `read`, `apply_patch`. Shell-first — `bash` is preferred
    ///   over dedicated `glob`/`grep`, and `apply_patch` is preferred over `write`+`edit`.
    /// * others: `bash`, `read`, `write`, `edit`, `glob`, `grep`.
    pub fn system_tools(mut self) -> Self {
        self.tools.extend(if self.model.starts_with("openai/") {
            vec![
                get_bash_tool_desc(),
                get_read_tool_desc(),
                get_apply_patch_tool_desc(),
            ]
        } else {
            vec![
                get_bash_tool_desc(),
                get_read_tool_desc(),
                get_write_tool_desc(),
                get_edit_tool_desc(),
                get_glob_tool_desc(),
                get_grep_tool_desc(),
            ]
        });
        self
    }

    pub fn python_repl_tool(mut self) -> Self {
        self.tools.push(get_python_repl_tool_desc());
        self
    }

    pub fn web_search_tool(mut self) -> Self {
        self.tools.push(get_web_search_tool_desc());
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
}
