use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentCard,
    lang_model::{LangModelOptions, model_family},
    tool::{
        ToolDesc, WebSearchEngineKind,
        r#impl::{
            get_apply_patch_tool_desc, get_edit_tool_desc, get_imgread_tool_desc,
            get_read_file_tool_desc, get_read_tool_desc, get_shell_tool_desc,
            get_web_fetch_tool_desc, get_web_search_tool_desc, get_write_tool_desc,
        },
    },
};

/// What makes an agent distinct: its model, instruction, tools and sub-agents.
///
/// Credentials and tool sources live on [`AgentProvider`](crate::agent::AgentProvider), the
/// [`ConsoleClient`](crate::console::ConsoleClient) on [`AgentState`](crate::agent::AgentState).
///
/// [`instruction`](AgentSpec::instruction) is private guidance to the model, never seen by
/// callers; [`card`](AgentSpec::card) is what a calling agent reads to decide whether to
/// delegate here. A sub-agent must have a card, since it names and describes the parent's
/// tool for it.
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDesc>,

    /// Sub-agents available to the agent (each registered as a callable tool)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subagents: Vec<AgentSpec>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_options: Option<LangModelOptions>,

    /// Public self-introduction to a calling agent; required when this spec is a sub-agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<AgentCard>,

    /// Engines used by the `web_search` tool. `None` (or not provided) means
    /// all available engines. Only meaningful when `web_search` is listed in `tools`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_engines: Option<Vec<WebSearchEngineKind>>,

    /// Skill directories in the console, each holding a `SKILL.md`. See
    /// [`skill`](Self::skill).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<String>,
}

impl AgentSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instruction: None,
            tools: Vec::new(),
            subagents: Vec::new(),
            card: None,
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
        self.tools.append(&mut tools.into_iter().collect());
        self
    }

    /// Append the canonical local-execution toolset for the spec's model family.
    pub fn system_tools(mut self) -> Self {
        self.tools.push(get_shell_tool_desc());

        let (family, name) = model_family(&self.model);
        // OpenAI models read files through `shell`, as in Codex.
        if family == "openai" {
            self.tools.push(get_apply_patch_tool_desc());
        } else {
            // DeepSeek, Kimi and GLM serve Anthropic-compatible APIs for Claude Code, so
            // likely follow the Claude style; Qwen Code is a fork of Gemini CLI.
            self.tools.push(match family {
                "anthropic" | "deepseek" | "moonshotai" | "z-ai" => get_read_tool_desc(),
                "google" | "qwen" => get_read_file_tool_desc(),
                _ => get_read_tool_desc(),
            });
            self.tools.push(get_write_tool_desc());
            self.tools.push(get_edit_tool_desc());
        }

        // Text-only models (no image input) get no `imgread`.
        let text_only = match family {
            "deepseek" => true,
            "moonshotai" => {
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

    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::new)
            .max_tokens = Some(max_tokens);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::new)
            .temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::new)
            .top_p = Some(top_p);
        self
    }

    pub fn top_k(mut self, top_k: u64) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::new)
            .top_k = Some(top_k);
        self
    }

    pub fn response_format(mut self, fmt: crate::lang_model::ResponseFormat) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::new)
            .response_format = Some(fmt);
        self
    }

    pub fn reasoning(mut self, effort: crate::lang_model::ReasoningEffort) -> Self {
        self.model_options
            .get_or_insert_with(LangModelOptions::new)
            .reasoning = Some(effort);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_tool_empty_engines_keeps_field_none() {
        let spec = AgentSpec::new("openai/gpt-4o-mini").web_search_tool(vec![]);
        assert!(
            spec.web_search_engines.is_none(),
            "empty engines should leave web_search_engines as None"
        );
        assert_eq!(spec.tools.len(), 1);
        assert_eq!(spec.tools[0].name, "web_search");
    }

    #[test]
    fn test_web_search_tool_with_engines_stores_field() {
        let spec = AgentSpec::new("openai/gpt-4o-mini").web_search_tool(vec![
            WebSearchEngineKind::Google,
            WebSearchEngineKind::Brave,
        ]);
        assert_eq!(
            spec.web_search_engines.as_deref(),
            Some(vec![WebSearchEngineKind::Google, WebSearchEngineKind::Brave].as_slice())
        );
        assert_eq!(spec.tools.len(), 1);
    }

    #[test]
    fn test_web_search_engines_omitted_from_serialisation_when_none() {
        let spec = AgentSpec::new("openai/gpt-4o-mini").web_search_tool(vec![]);
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            !json.contains("web_search_engines"),
            "web_search_engines should be absent when None: {json}"
        );
    }

    #[test]
    fn test_web_search_engines_roundtrip() {
        let spec = AgentSpec::new("openai/gpt-4o-mini").web_search_tool(vec![
            WebSearchEngineKind::Google,
            WebSearchEngineKind::Yahoo,
        ]);
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.web_search_engines.as_deref(),
            Some(vec![WebSearchEngineKind::Google, WebSearchEngineKind::Yahoo].as_slice())
        );
    }
}
