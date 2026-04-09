use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

/// Inference-time parameters forwarded to the language model on each call.
///
/// All fields are optional. When a field is `None`, provider-specific defaults apply
/// (e.g. Anthropic defaults `max_tokens` to 8192 because it is a required API field).
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LangModelInferConfig {
    /// Maximum number of tokens the model may generate in a single response.
    pub max_tokens: Option<u64>,
}

/// Defines the logical identity of an agent as configured by the user.
///
/// `AgentSpec` captures what makes an agent distinct — the language model it uses,
/// the system instruction that shapes its behavior, and the set of tools it has access to.
/// Changing any of these fields changes the fundamental nature of the agent.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpec {
    /// Identifier of the language model (e.g. `"claude-sonnet-4-6"`)
    pub lm: String,

    /// Optional system prompt that guides the agent's behavior
    pub instruction: Option<String>,

    /// Names of tools available to the agent
    pub tools: Vec<String>,
}

impl AgentSpec {
    pub fn new(lm: impl Into<String>) -> Self {
        Self {
            lm: lm.into(),
            instruction: None,
            tools: vec![],
        }
    }

    pub fn with_instruction(mut self, inst: String) -> Self {
        self.instruction = Some(inst);
        self
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = tools.into_iter().map(|v| v.into()).collect();
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
