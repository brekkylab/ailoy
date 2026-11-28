use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    model::{LangModelOptions, ModelKeyProvider, ThinkingEffort},
};

/// What makes an agent distinct: its model, instruction and tools.
///
/// It names what the agent uses and holds none of it: where the agent runs, the model is built
/// from its desc and each tool from its name. The sandbox is the state's; the spec says whether
/// the agent uses it, and the browser in it.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpec {
    /// The model's desc, `<org>` or `<org>/<name>` (e.g. `"anthropic/claude-sonnet-5-5"`).
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_options: Option<LangModelOptions>,

    /// System prompt that shapes how the model works.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    /// The names of the tools exposed to the model. Where the agent runs has to provide each:
    /// its [`ToolProvider`](crate::tool::ToolProvider) describes it to the model and makes it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,

    /// Skill directories in the sandbox, each holding a `SKILL.md`. See
    /// [`skill`](Self::skill).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<String>,

    /// Whether the agent works in a sandbox, which its state then has to hold. It brings a
    /// shell and file tools there, in the style of the model's org, but for any `tools` names
    /// itself.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub use_sandbox: bool,

    /// Whether the agent drives the browser of its sandbox, virtx's `browser` abin, which the
    /// sandbox then has to have. It brings the `browser` and `browser_screenshot` tools.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub use_browser: bool,

    /// How old tool results are truncated in the history as each turn begins.
    #[serde(default)]
    pub truncation: TruncationPolicy,
}

impl AgentSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_options: None,
            instruction: None,
            tools: Vec::new(),
            skills: Vec::new(),
            use_sandbox: false,
            use_browser: false,
            truncation: TruncationPolicy::default(),
        }
    }

    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.instruction = Some(inst.into());
        self
    }

    /// Give this agent the skill in `dir`, a directory in the sandbox holding a `SKILL.md`.
    ///
    /// At construction the frontmatter `name` and `description` are read through the
    /// console and listed in the system message with the skill's location; the agent reads
    /// the rest itself, so it needs a console and a file-reading tool.
    ///
    /// Skills are added only to the instruction-built system message, never to one already
    /// in the history.
    pub fn skill(mut self, dir: impl Into<String>) -> Self {
        self.skills.push(dir.into());
        self
    }

    /// Truncate old tool results in the history by `policy`. See [`TruncationPolicy`].
    pub fn truncation(mut self, policy: TruncationPolicy) -> Self {
        self.truncation = policy;
        self
    }

    pub fn tool(mut self, name: impl Into<String>) -> Self {
        self.tools.push(name.into());
        self
    }

    pub fn tools(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools.extend(names.into_iter().map(Into::into));
        self
    }

    /// Work in a sandbox, which the agent's state then has to hold. See
    /// [`use_sandbox`](Self::use_sandbox).
    pub fn sandbox(mut self) -> Self {
        self.use_sandbox = true;
        self
    }

    /// Drive the browser of the sandbox. See [`use_browser`](Self::use_browser).
    pub fn browser(mut self) -> Self {
        self.use_browser = true;
        self
    }

    /// Add `web_search`, unless the spec has it. How it searches is set where the agent runs.
    pub fn web_search_tool(self) -> Self {
        self.tool_once("web_search")
    }

    /// Add `web_fetch`, unless the spec has it: one `url` per call, long bodies paged through
    /// `offset`, at most one request per second per host.
    pub fn web_fetch_tool(self) -> Self {
        self.tool_once("web_fetch")
    }

    fn tool_once(mut self, name: &str) -> Self {
        if !self.tools.iter().any(|t| t == name) {
            self.tools.push(name.to_owned());
        }
        self
    }

    pub fn thinking_effort(mut self, effort: ThinkingEffort) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::default)
            .thinking_effort = Some(effort);
        self
    }

    /// JSON schema the model's answers must match.
    pub fn output_schema(mut self, schema: Value) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::default)
            .output_schema = Some(schema);
        self
    }
}

/// What keeping an agent on a server takes: `PUT /agents/{name}`'s body.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentPutArgs {
    /// What the agent is. The server has to provide the model and each tool it names.
    pub spec: AgentSpec,

    /// Which end provides the key the agent's model is called with.
    #[serde(default)]
    pub model_key_provider: ModelKeyProvider,
}

impl AgentPutArgs {
    /// `spec`, its model called with the server's key.
    pub fn new(spec: AgentSpec) -> Self {
        Self {
            spec,
            model_key_provider: ModelKeyProvider::default(),
        }
    }

    pub fn model_key_provider(mut self, provider: ModelKeyProvider) -> Self {
        self.model_key_provider = provider;
        self
    }
}

/// An agent a server keeps, as the server answers it.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentListOutputItem {
    /// The name the agent is kept under, and run by.
    pub name: String,

    /// What the agent is.
    pub spec: AgentSpec,

    /// Which end provides the key the agent's model is called with.
    pub model_key_provider: ModelKeyProvider,
}

/// How old tool results in an agent's history are replaced by placeholders as each turn
/// begins, so a long run stays within the model's context window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct TruncationPolicy {
    /// The input tokens of the last model call past which old tool results are replaced.
    /// `0`, the default, turns this off.
    pub max_input_tokens: u64,
    /// The latest user turns whose tool results are left as they are. The turn just begun
    /// is always one of them, so `0` acts as `1`.
    pub preserve_recent_turns: usize,
    /// Images in tool results the history may hold; past it, the oldest are replaced.
    /// `0` turns this off.
    pub max_images: usize,
    /// Images in tool results left once `max_images` is passed.
    pub preserve_recent_images: usize,
}

impl Default for TruncationPolicy {
    fn default() -> Self {
        Self {
            max_input_tokens: 0,
            preserve_recent_turns: 3,
            max_images: 30,
            preserve_recent_images: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_spec_roundtrip() {
        let spec = AgentSpec::new("anthropic/claude-sonnet-5-5")
            .instruction("Be brief.")
            .thinking_effort(ThinkingEffort::Low)
            .tool("web_search")
            .truncation(TruncationPolicy {
                max_input_tokens: 50_000,
                ..Default::default()
            });
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "anthropic/claude-sonnet-5-5");
        assert_eq!(
            back.model_options.as_ref().unwrap().thinking_effort,
            Some(ThinkingEffort::Low)
        );
        assert_eq!(back.tools, ["web_search"]);
        assert_eq!(back.truncation.max_input_tokens, 50_000);
    }

    #[test]
    fn test_the_key_is_the_server_s_unless_said() {
        let args: AgentPutArgs =
            serde_json::from_str(r#"{"spec": {"model": "anthropic"}}"#).unwrap();
        assert_eq!(args.model_key_provider, ModelKeyProvider::Server);
    }

    #[test]
    fn test_web_tools_are_added_once() {
        let spec = AgentSpec::new("anthropic")
            .web_search_tool()
            .web_fetch_tool()
            .web_search_tool();
        assert_eq!(spec.tools, ["web_search", "web_fetch"]);
    }

    /// A spec saved without `truncation` loads with the default policy.
    #[test]
    fn test_spec_without_truncation_takes_the_default() {
        let spec: AgentSpec = serde_json::from_str(r#"{"model": "anthropic"}"#).unwrap();
        assert_eq!(spec.truncation, TruncationPolicy::default());
    }
}
