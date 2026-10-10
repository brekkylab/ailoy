use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    experimental::model::{LangModelOptions, ThinkingEffort},
    tool::{
        ToolDesc, WebSearchEngineKind,
        r#impl::{
            get_apply_patch_tool_desc, get_edit_tool_desc, get_imgread_tool_desc,
            get_read_file_tool_desc, get_read_tool_desc, get_shell_tool_desc,
            get_web_fetch_tool_desc, get_web_search_tool_desc, get_write_tool_desc,
        },
    },
};

/// What makes an agent distinct: its model, instruction and tools.
///
/// Tool sources live on [`AgentProvider`](super::AgentProvider), the console on
/// [`AgentState`](super::AgentState).
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpec {
    /// The model's desc, as [`create_lang_model`](crate::experimental::model::create_lang_model)
    /// takes it (e.g. `"claude/sonnet"`).
    pub model: String,

    /// System prompt that shapes how the model works.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    /// Tool descriptions exposed to the model. Each [`ToolDesc::name`] must match
    /// an entry registered in the [`AgentProvider`](super::AgentProvider)'s
    /// [`ToolProvider`](crate::tool::ToolProvider).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDesc>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_options: Option<LangModelOptions>,

    /// Engines used by the `web_search` tool. `None` (or not provided) means
    /// all available engines. Only meaningful when `web_search` is listed in `tools`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_engines: Option<Vec<WebSearchEngineKind>>,

    /// Skill directories in the console, each holding a `SKILL.md`. See
    /// [`skill`](Self::skill).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<String>,
}

/// The model a desc names (`"claude"` of `"claude/sonnet"`) and the name after it.
pub(super) fn split_desc(desc: &str) -> (&str, &str) {
    desc.split_once('/').unwrap_or((desc, ""))
}

impl AgentSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instruction: None,
            tools: Vec::new(),
            model_options: None,
            web_search_engines: None,
            skills: Vec::new(),
        }
    }

    pub fn instruction(mut self, inst: impl Into<String>) -> Self {
        self.instruction = Some(inst.into());
        self
    }

    /// Give this agent the skill in `dir`, a directory in the console holding a `SKILL.md`.
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

    pub fn tool(mut self, tool: ToolDesc) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDesc>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Append the canonical local-execution toolset for the spec's model.
    pub fn system_tools(mut self) -> Self {
        self.tools.push(get_shell_tool_desc());

        let (model, name) = split_desc(&self.model);
        // GPT reads files through `shell`, as in Codex.
        if model == "gpt" {
            self.tools.push(get_apply_patch_tool_desc());
        } else {
            // DeepSeek, Kimi and GLM serve Anthropic-compatible APIs for Claude Code, so
            // likely follow the Claude style.
            self.tools.push(match model {
                "gemini" => get_read_file_tool_desc(),
                _ => get_read_tool_desc(),
            });
            self.tools.push(get_write_tool_desc());
            self.tools.push(get_edit_tool_desc());
        }

        // Text-only models (no image input) get no `imgread`. The name may carry a
        // backend's vendor prefix (`moonshotai/…`, `moonshot.…`).
        let name = name.rsplit(['/', '.']).next().unwrap_or(name);
        let text_only = match model {
            "deepseek" => true,
            "kimi" => {
                name.starts_with("kimi-k2-")
                    || (name.starts_with("moonshot-v1-") && !name.contains("vision"))
            }
            _ => false,
        };
        if !text_only {
            self.tools.push(get_imgread_tool_desc());
        }

        self
    }

    /// Append only the `shell` tool, without the other `system_tools` entries.
    pub fn shell_tool(mut self) -> Self {
        self.tools.push(get_shell_tool_desc());
        self
    }

    /// Add `web_search`; a non-empty `engines` restricts it to those, empty uses all.
    pub fn web_search_tool(mut self, engines: Vec<WebSearchEngineKind>) -> Self {
        self.tools.push(get_web_search_tool_desc());
        if !engines.is_empty() {
            self.web_search_engines = Some(engines);
        }
        self
    }

    /// Add the `web_fetch` tool to the spec.
    ///
    /// Not part of `system_tools()`. The tool fetches one `url` per call, pages long bodies
    /// through `offset`, and allows one request per second per host.
    pub fn web_fetch_tool(mut self) -> Self {
        self.tools.push(get_web_fetch_tool_desc());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn names(spec: &AgentSpec) -> Vec<&str> {
        spec.tools.iter().map(|t| t.name.as_str()).collect()
    }

    #[test]
    fn test_system_tools_follow_the_model() {
        let gpt = AgentSpec::new("gpt/gpt-6-sol").system_tools();
        assert_eq!(names(&gpt), ["shell", "apply_patch", "imgread"]);

        let gemini = AgentSpec::new("gemini").system_tools();
        assert!(
            names(&gemini).contains(&"read_file"),
            "{:?}",
            names(&gemini)
        );

        let claude = AgentSpec::new("claude/sonnet").system_tools();
        assert!(names(&claude).contains(&"read"), "{:?}", names(&claude));
        assert!(names(&claude).contains(&"imgread"), "{:?}", names(&claude));

        let deepseek = AgentSpec::new("deepseek/deepseek-chat").system_tools();
        assert!(!names(&deepseek).contains(&"imgread"));

        // Through OpenRouter the name carries the vendor.
        let kimi = AgentSpec::new("kimi/moonshotai/kimi-k2-0905").system_tools();
        assert!(!names(&kimi).contains(&"imgread"));
    }

    #[test]
    fn test_web_search_tool_empty_engines_keeps_field_none() {
        let spec = AgentSpec::new("claude").web_search_tool(vec![]);
        assert!(spec.web_search_engines.is_none());
        assert_eq!(names(&spec), ["web_search"]);
    }

    #[test]
    fn test_spec_roundtrip() {
        let spec = AgentSpec::new("claude/sonnet")
            .instruction("Be brief.")
            .thinking_effort(ThinkingEffort::Low)
            .web_search_tool(vec![WebSearchEngineKind::Google]);
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "claude/sonnet");
        assert_eq!(
            back.model_options.unwrap().thinking_effort,
            Some(ThinkingEffort::Low)
        );
        assert_eq!(
            back.web_search_engines.as_deref(),
            Some([WebSearchEngineKind::Google].as_slice())
        );
    }
}
