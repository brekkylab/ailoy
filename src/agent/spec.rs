use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    lang_model::{LangModelAPISchema, LangModelProvider},
    message::AgentCard,
    tool::{BuiltinToolProvider, MCPToolProvider, ToolProvider},
};

/// Defines the logical identity of an agent as configured by the user.
///
/// `AgentSpec` captures what makes an agent distinct — the language model it uses,
/// the system instruction that shapes its behavior, and the set of tools it has access to.
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

    /// Public self-introduction exposed to a calling agent or orchestrator.
    ///
    /// Only relevant when this agent acts as a sub-agent.
    /// `None` for top-level agents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<AgentCard>,
}

impl AgentSpec {
    pub fn new(lm: impl Into<String>) -> Self {
        Self {
            model: lm.into(),
            instruction: None,
            tools: Vec::new(),
            card: None,
        }
    }

    pub fn instruction(mut self, inst: String) -> Self {
        self.instruction = Some(inst);
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

    pub fn card(mut self, card: AgentCard) -> Self {
        self.card = Some(card);
        self
    }
}

/// Global registry of runtime resources available to all agents.
///
/// `AgentProvider` is a **shared, top-level configuration object** that is passed
/// unchanged to every agent in a session.  It answers two questions:
///
/// * **How do I call a model?** — `models` maps every model name that any
///   [`AgentSpec`] might reference to its API schema, endpoint URL, and
///   credentials.  An agent looks up its [`AgentSpec::model`] here at startup.
///
/// * **How do I initialise a tool?** — `tools` lists the tool-source configurations
///   (built-in tool settings and MCP server connections) that are spun up to
///   populate the [`ToolSet`] before an agent starts.  Each entry can contribute
///   one or more runnable tools.
///
/// `AgentProvider` is separate from [`AgentSpec`] because these settings describe
/// *how* to run an agent, not *what* the agent is.  Swapping the API endpoint or
/// key does not change the agent's identity; swapping the model or instruction does.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentProvider {
    /// Registry of all available language models.
    ///
    /// Keys are model names (e.g. `"claude-sonnet-4-6"`) that match the
    /// [`AgentSpec::model`] field.  Values describe how to reach the model:
    /// API wire format, endpoint URL, and optional credentials.
    pub models: BTreeMap<String, LangModelProvider>,

    /// Toolset initialisation configs.
    ///
    /// Each entry describes how to spin up a tool source — either a built-in
    /// tool bundled with the runtime (e.g. `WebSearch`, `PythonRepl`) or an
    /// external MCP server (stdio child-process or HTTP stream).  All sources
    /// are initialised at agent startup and merged into the agent's [`ToolSet`].
    pub tools: Vec<ToolProvider>,
}

impl AgentProvider {
    pub fn new() -> Self {
        Self {
            models: Default::default(),
            tools: Default::default(),
        }
    }

    /// Register a model name or glob pattern with an explicit provider.
    pub fn model(mut self, pattern: impl Into<String>, provider: LangModelProvider) -> Self {
        self.models.insert(pattern.into(), provider);
        self
    }

    /// Register all OpenAI models under the `openai/*` namespace via the Responses API.
    ///
    /// Use model names like `"openai/gpt-4o"` or `"openai/o3-mini"` in [`AgentSpec`].
    pub fn model_openai(mut self, api_key: String) -> Self {
        self.models.insert(
            "openai/*".to_string(),
            LangModelProvider::API {
                schema: LangModelAPISchema::ChatCompletion,
                url: Url::parse("https://api.openai.com/v1/chat/completions").unwrap(),
                api_key: Some(api_key),
                max_tokens: None,
            },
        );
        self
    }

    /// Register all Claude models under the `anthropic/*` namespace via the Messages API.
    ///
    /// Use model names like `"anthropic/claude-sonnet-4-6"` in [`AgentSpec`].
    pub fn model_claude(mut self, api_key: String) -> Self {
        self.models.insert(
            "anthropic/claude-*".to_string(),
            LangModelProvider::API {
                schema: LangModelAPISchema::Anthropic,
                url: Url::parse("https://api.anthropic.com/v1/messages").unwrap(),
                api_key: Some(api_key),
                max_tokens: None,
            },
        );
        self
    }

    /// Register all Gemini models under the `google/*` namespace via the Generative Language API.
    ///
    /// Use model names like `"google/gemini-2.0-flash"` in [`AgentSpec`].
    pub fn model_gemini(mut self, api_key: String) -> Self {
        self.models.insert(
            "google/gemini-*".to_string(),
            LangModelProvider::API {
                schema: LangModelAPISchema::Gemini,
                url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/")
                    .unwrap(),
                api_key: Some(api_key),
                max_tokens: None,
            },
        );
        self
    }

    /// Resolve a model name to its provider.
    /// (e.g. `"openai/gpt-4o"` → `"gpt-4o"`).
    pub fn get_model(&self, model: impl AsRef<str>) -> Option<&LangModelProvider> {
        fn glob_match(pattern: &str, text: &str) -> bool {
            let p: Vec<char> = pattern.chars().collect();
            let t: Vec<char> = text.chars().collect();
            glob_match_chars(&p, &t)
        }

        fn glob_match_chars(p: &[char], t: &[char]) -> bool {
            match (p.split_first(), t.split_first()) {
                (None, None) => true,
                (None, Some(_)) => false,
                (Some((&'*', rest_p)), _) => {
                    // * matches zero characters here, or consume one character from text
                    glob_match_chars(rest_p, t)
                        || t.split_first()
                            .is_some_and(|(_, rest_t)| glob_match_chars(p, rest_t))
                }
                (Some((&'?', rest_p)), Some((_, rest_t))) => glob_match_chars(rest_p, rest_t),
                (Some((pc, rest_p)), Some((tc, rest_t))) if pc == tc => {
                    glob_match_chars(rest_p, rest_t)
                }
                _ => false,
            }
        }

        let model = model.as_ref();
        if let Some(p) = self.models.get(model) {
            return Some(p);
        }
        self.models
            .iter()
            .filter(|(pattern, _)| glob_match(pattern, model))
            .max_by_key(|(pattern, _)| pattern.chars().filter(|&c| c != '*' && c != '?').count())
            .map(|(_, provider)| provider)
    }

    /// Register a tool.
    pub fn tool(mut self, tool: ToolProvider) -> Self {
        self.tools.push(tool);
        self
    }

    /// Register tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolProvider>) -> Self {
        for tool in tools.into_iter() {
            self.tools.push(tool);
        }
        self
    }

    /// Register the built-in web search tool.
    pub fn tool_web_search(mut self) -> Self {
        self.tools
            .push(ToolProvider::Builtin(BuiltinToolProvider::WebSearch {}));
        self
    }

    /// Register a sub-agent as a tool.
    ///
    /// The `spec` must have a [`card`](AgentSpec::card) set — the card's `name`
    /// becomes the tool name that the orchestrating agent calls, and its
    /// `description` is shown to the model to help it decide when to delegate.
    pub fn tool_sub_agent(mut self, spec: AgentSpec) -> Self {
        self.tools.push(ToolProvider::SubAgent(spec));
        self
    }

    /// Register an MCP tool server reachable via a child-process stdio connection.
    pub fn tool_mcp_stdio(mut self, command: impl Into<String>) -> Self {
        self.tools.push(ToolProvider::MCP(MCPToolProvider::Stdio {
            command: command.into(),
        }));
        self
    }

    /// Register an MCP tool server reachable via HTTP streaming.
    pub fn tool_mcp_streamable_http(mut self, url: Url) -> Self {
        self.tools
            .push(ToolProvider::MCP(MCPToolProvider::StreamableHTTP { url }));
        self
    }

    /// Register a remote A2A agent as a tool.
    ///
    /// The runtime fetches `{url}/.well-known/agent-card.json` at startup to
    /// discover the agent's name and description.
    pub fn tool_a2a(mut self, url: Url) -> Self {
        self.tools.push(ToolProvider::A2A { url });
        self
    }

    /// Get tools
    pub fn get_tools(&self) -> &[ToolProvider] {
        &self.tools
    }
}

impl Default for AgentProvider {
    fn default() -> Self {
        Self::new()
    }
}
