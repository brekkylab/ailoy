use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentSpec {
    pub lm: String,
    pub instruction: Option<String>,
    pub tools: Vec<String>,
}

impl AgentSpec {
    pub fn new(lm: String) -> Self {
        Self {
            lm,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LangModelAPISchema {
    ChatCompletion,
    Anthropic,
}

/// Specifies how a language model is executed
///
/// This describes the runtime provider required to actually run the model
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LangModelProvider {
    API {
        schema: LangModelAPISchema,

        url: Url,

        api_key: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCPToolProvider {
    Stdin { command: String },
    StreamableHTTP { url: Url },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolProvider {
    Builtin { name: String },
    MCP(MCPToolProvider),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentProvider {
    pub lm: LangModelProvider,
    pub tools: Vec<ToolProvider>,
}
