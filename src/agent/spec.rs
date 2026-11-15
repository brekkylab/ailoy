use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{Message, Part, Role},
    model::{LangModelOptions, ThinkingEffort},
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
/// Tools are built through the providers set with
/// [`set_tool_provider`](crate::tool::set_tool_provider); the console lives on
/// [`AgentState`](super::AgentState).
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpec {
    /// The model's desc, as [`create_lang_model`](crate::model::create_lang_model)
    /// takes it (e.g. `"anthropic/claude-sonnet-5-5"`).
    pub model: String,

    /// System prompt that shapes how the model works.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    /// Tool descriptions exposed to the model. Each [`ToolDesc::name`] must have a
    /// [`ToolProvider`](crate::tool::ToolProvider) set for it.
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

    /// How old tool results are truncated in the history as each turn begins.
    #[serde(default)]
    pub truncation: TruncationPolicy,
}

/// The org a desc names (`"anthropic"` of `"anthropic/claude-sonnet-5-5"`) and the name
/// after it.
pub(super) fn split_desc(desc: &str) -> (&str, &str) {
    desc.split_once('/').unwrap_or((desc, ""))
}

/// The shell and file tools for a sandbox, in the style of the model's org.
fn sandbox_tool_descs(org: &str) -> Vec<ToolDesc> {
    let mut descs = vec![get_shell_tool_desc()];
    // GPT reads files through `shell`, as in Codex.
    if org == "openai" {
        descs.push(get_apply_patch_tool_desc());
    } else {
        // DeepSeek, Kimi and GLM serve Anthropic-compatible APIs for Claude Code, so
        // likely follow the Claude style.
        descs.push(match org {
            "google" => get_read_file_tool_desc(),
            _ => get_read_tool_desc(),
        });
        descs.push(get_write_tool_desc());
        descs.push(get_edit_tool_desc());
    }

    // `imgread` only where the image reaches the model: the backends of these take tool
    // results as text alone (an image becomes a label), even for vision models.
    if !matches!(
        org,
        "deepseek-ai" | "moonshotai" | "zai-org" | "Qwen" | "LGAI-EXAONE"
    ) {
        descs.push(get_imgread_tool_desc());
    }
    descs
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
            truncation: TruncationPolicy::default(),
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

    /// Truncate old tool results in the history by `policy`. See [`TruncationPolicy`].
    pub fn truncation(mut self, policy: TruncationPolicy) -> Self {
        self.truncation = policy;
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

    /// Add the shell and file tools for a sandbox, in the style of the model's org, skipping
    /// any the spec already has. [`AgentBuilder::sandbox`](super::AgentBuilder::sandbox) adds
    /// them itself.
    pub fn sandbox_tools(mut self) -> Self {
        for desc in sandbox_tool_descs(split_desc(&self.model).0) {
            if !self.tools.iter().any(|t| t.name == desc.name) {
                self.tools.push(desc);
            }
        }
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
    /// The tool fetches one `url` per call, pages long bodies
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

/// What stands in for an image [`TruncationPolicy::apply`] replaced.
pub(super) const IMAGE_PLACEHOLDER: &str = "[image from an earlier turn removed]";

/// What stands in for a tool result's contents [`TruncationPolicy::apply`] replaced.
pub(super) const TOOL_RESULT_PLACEHOLDER: &str = "[context truncated]";

impl TruncationPolicy {
    /// Replace old images and tool results in `history` by placeholders. No message is
    /// removed, and a tool result keeps its id, since APIs such as Claude's refuse a call
    /// without its result. Images a person attached are left alone.
    ///
    /// Only for the start of a turn, with the query already in `history`: Claude refuses a
    /// thinking block handed back once anything before it has changed, and at turn start
    /// none follows, since the backends replay thinking only after the last user message.
    pub(super) fn apply(&self, history: &mut [Message], last_input_tokens: Option<u64>) {
        // Past `max_images` images in tool results, the oldest go until
        // `preserve_recent_images` remain: many at once, so the prompt cache breaks once in
        // several turns rather than every turn.
        if self.max_images > 0 {
            let images: Vec<(usize, usize)> = history
                .iter()
                .enumerate()
                .filter(|(_, m)| m.role == Role::Tool)
                .flat_map(|(i, m)| {
                    let parts = m.contents.iter().enumerate();
                    parts
                        .filter(|(_, p)| p.is_image())
                        .map(move |(j, _)| (i, j))
                })
                .collect();
            if images.len() > self.max_images {
                let replaced = images.len().saturating_sub(self.preserve_recent_images);
                for (i, j) in images.into_iter().take(replaced) {
                    history[i].contents[j] = Part::text(IMAGE_PLACEHOLDER);
                }
            }
        }

        // Once the last call read past `max_input_tokens`, the tool results before the
        // latest `preserve_recent_turns` user turns lose their contents. A turn counts from
        // its user message, since one turn may hold many assistant messages; with fewer turns
        // than that, nothing changes.
        if self.max_input_tokens == 0 || last_input_tokens.unwrap_or(0) <= self.max_input_tokens {
            return;
        }
        let users = history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::User);
        let Some((preserve_from, _)) = users.rev().nth(self.preserve_recent_turns.max(1) - 1)
        else {
            return;
        };
        for message in &mut history[..preserve_from] {
            if message.role == Role::Tool {
                message.contents = vec![Part::text(TOOL_RESULT_PLACEHOLDER)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys() -> Message {
        Message::new(Role::System).with_contents([Part::text("system")])
    }

    fn user(text: &str) -> Message {
        Message::new(Role::User).with_contents([Part::text(text)])
    }

    fn asst(text: &str) -> Message {
        Message::new(Role::Assistant).with_contents([Part::text(text)])
    }

    fn call(id: &str) -> Message {
        Message::new(Role::Assistant).with_tool_calls([Part::function(id, "echo", Value::null())])
    }

    fn result(id: &str) -> Message {
        Message::new(Role::Tool)
            .with_id(id)
            .with_contents([Part::text("ok")])
    }

    fn image() -> Part {
        Part::image_embedded("image/png", vec![0u8; 4].into()).unwrap()
    }

    fn shot(id: &str) -> Message {
        Message::new(Role::Tool)
            .with_id(id)
            .with_contents([Part::text("the page"), image()])
    }

    fn images_in(history: &[Message]) -> usize {
        history
            .iter()
            .flat_map(|m| &m.contents)
            .filter(|p| p.is_image())
            .count()
    }

    /// Whether each tool result in `history` lost its contents.
    fn truncated(history: &[Message]) -> Vec<bool> {
        history
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.contents[0].as_text() == Some(TOOL_RESULT_PLACEHOLDER))
            .collect()
    }

    /// The tool results of the turns before the latest `preserve_recent_turns` lose their
    /// contents and keep their ids; no message goes. Turns count from user messages, so a
    /// turn of several assistant messages is one.
    #[test]
    fn test_old_tool_results_are_truncated_past_the_token_limit() {
        let history = vec![
            sys(),
            user("u1"),
            call("c1"),
            result("c1"),
            asst("a1"),
            user("u2"),
            call("c2"),
            result("c2"),
            asst("a2"),
            user("u3"),
        ];
        let policy = |preserve_recent_turns| TruncationPolicy {
            max_input_tokens: 100,
            preserve_recent_turns,
            ..Default::default()
        };

        let mut cut = history.clone();
        policy(2).apply(&mut cut, Some(200));
        assert_eq!(truncated(&cut), [true, false]);
        assert_eq!(cut.len(), history.len());
        assert_eq!(cut[0].role, Role::System);
        assert_eq!(cut[3].id.as_deref(), Some("c1"));

        // The turn just begun is always kept, so `0` acts as `1`.
        for preserve in [0, 1] {
            let mut cut = history.clone();
            policy(preserve).apply(&mut cut, Some(200));
            assert_eq!(truncated(&cut), [true, true], "preserve: {preserve}");
        }

        // Fewer turns than preserved, at the limit, or without a limit: nothing changes.
        for (preserve, max_input_tokens, read) in [(3, 100, 200), (1, 100, 100), (1, 0, 200)] {
            let mut cut = history.clone();
            TruncationPolicy {
                max_input_tokens,
                preserve_recent_turns: preserve,
                ..Default::default()
            }
            .apply(&mut cut, Some(read));
            assert_eq!(
                truncated(&cut),
                [false, false],
                "{preserve} {max_input_tokens} {read}"
            );
        }
    }

    /// Past the cap the oldest images in tool results go until the preserved remain; at it,
    /// none does. The text beside each and a person's images stay.
    #[test]
    fn test_images_past_the_cap_are_replaced_oldest_first() {
        let images = |max_images, preserve_recent_images| TruncationPolicy {
            max_images,
            preserve_recent_images,
            ..Default::default()
        };
        let mut history = vec![
            sys(),
            Message::new(Role::User).with_contents([Part::text("u1"), image()]),
        ];
        history.extend((1..=4).map(|n| shot(&format!("c{n}"))));

        images(4, 2).apply(&mut history, None);
        assert_eq!(images_in(&history), 5);

        history.push(shot("c5"));
        images(4, 2).apply(&mut history, None);
        assert_eq!(images_in(&history), 3);
        assert!(history[1].contents[1].is_image());
        assert!(
            history[2..=4]
                .iter()
                .all(|m| m.contents[1].as_text() == Some(IMAGE_PLACEHOLDER))
        );
        assert!(history[5..=6].iter().all(|m| m.contents[1].is_image()));
        assert_eq!(history[2].contents[0].as_text(), Some("the page"));
        assert_eq!(history[2].id.as_deref(), Some("c1"));

        // A cap of `0` is off; preserving none clears them all.
        images(0, 0).apply(&mut history, None);
        assert_eq!(images_in(&history), 3);
        images(1, 0).apply(&mut history, None);
        assert_eq!(images_in(&history), 1);
    }

    fn names(spec: &AgentSpec) -> Vec<&str> {
        spec.tools.iter().map(|t| t.name.as_str()).collect()
    }

    fn system_names(org: &str) -> Vec<String> {
        sandbox_tool_descs(org)
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    #[test]
    fn test_sandbox_tools_follow_the_model() {
        assert_eq!(system_names("openai"), ["shell", "apply_patch", "imgread"]);
        assert!(system_names("google").contains(&"read_file".into()));

        let claude = system_names("anthropic");
        assert!(claude.contains(&"read".into()), "{claude:?}");
        assert!(claude.contains(&"imgread".into()), "{claude:?}");

        assert!(!system_names("deepseek-ai").contains(&"imgread".into()));
        assert!(!system_names("moonshotai").contains(&"imgread".into()));
        assert!(!system_names("LGAI-EXAONE").contains(&"imgread".into()));
    }

    #[test]
    fn test_sandbox_tools_skip_tools_already_there() {
        let spec = AgentSpec::new("openai/gpt-6-luna")
            .sandbox_tools()
            .sandbox_tools();
        assert_eq!(names(&spec), ["shell", "apply_patch", "imgread"]);
    }

    #[test]
    fn test_web_search_tool_empty_engines_keeps_field_none() {
        let spec = AgentSpec::new("anthropic").web_search_tool(vec![]);
        assert!(spec.web_search_engines.is_none());
        assert_eq!(names(&spec), ["web_search"]);
    }

    #[test]
    fn test_spec_roundtrip() {
        let spec = AgentSpec::new("anthropic/claude-sonnet-5-5")
            .instruction("Be brief.")
            .thinking_effort(ThinkingEffort::Low)
            .web_search_tool(vec![WebSearchEngineKind::Google])
            .truncation(TruncationPolicy {
                max_input_tokens: 50_000,
                ..Default::default()
            });
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "anthropic/claude-sonnet-5-5");
        assert_eq!(
            back.model_options.unwrap().thinking_effort,
            Some(ThinkingEffort::Low)
        );
        assert_eq!(
            back.web_search_engines.as_deref(),
            Some([WebSearchEngineKind::Google].as_slice())
        );
        assert_eq!(back.truncation.max_input_tokens, 50_000);
    }

    /// A spec saved without `truncation` loads with the default policy.
    #[test]
    fn test_spec_without_truncation_takes_the_default() {
        let spec: AgentSpec = serde_json::from_str(r#"{"model": "anthropic"}"#).unwrap();
        assert_eq!(spec.truncation, TruncationPolicy::default());
    }
}
