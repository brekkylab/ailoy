use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::agent::AgentCard;

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
}

/// Wire protocol used when calling a language model API.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LangModelAPISchema {
    /// OpenAI-compatible `/v1/chat/completions` format
    ChatCompletion,

    /// Anthropic Messages API format
    Anthropic,

    /// Google Gemini API format
    Gemini,

    /// OpenAI Responses API format
    #[serde(rename = "openai")]
    OpenAI,
}

/// Describes the runtime endpoint used to invoke a language model.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LangModelProvider {
    /// Calls a remote HTTP API. Requires the wire `schema`, the `url` of the endpoint, and an optional `api_key` for authentication.
    API {
        schema: LangModelAPISchema,

        url: Url,

        api_key: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuiltinToolProvider {
    WebSearch {},
    ConvertPdfToMd {},
    PythonRepl {
        /// Python version to provision (e.g. `"3.12"`). `None` → latest stable.
        python_version: Option<String>,
        /// Persistent venv path. `None` → temp dir cleaned up when the tool is dropped.
        /// Supports `~` expansion.
        venv_path: Option<String>,
        /// Packages to pre-install before the first tool call.
        #[serde(default)]
        packages: Vec<String>,
    },
}

/// Transport configuration for an MCP (Model Context Protocol) tool server.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCPToolProvider {
    /// Spawns a child process and communicates over its stdio
    Stdio { command: String },

    /// Connects to a remote MCP server over HTTP streaming
    StreamableHTTP { url: Url },
}

/// Identifies where a tool's implementation lives at runtime.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolProvider {
    /// A tool baked into the agent runtime, referenced by `name`
    Builtin(BuiltinToolProvider),

    /// A tool served by an external MCP server described by [`MCPToolProvider`]
    MCP(MCPToolProvider),
}

/// Supplies the runtime parameters needed to execute an agent.
///
/// `AgentProvider` is separate from [`AgentSpec`] because these settings describe *how*
/// to run an agent, not *what* the agent is. Swapping the API endpoint or key does not
/// change the agent's identity; swapping the model or instruction does.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentProvider {
    /// The concrete language model provider (API schema, endpoint URL, credentials)
    pub lm: LangModelProvider,

    /// Resolved tool providers that back each tool name declared in [`AgentSpec::tools`]
    pub tools: Vec<ToolProvider>,
}
