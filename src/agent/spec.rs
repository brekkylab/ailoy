use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentCard,
    lang_model::LangModelOptions,
    runenv::FileEntry,
    tool::{
        ToolDesc, WebSearchEngineKind,
        r#impl::{
            get_apply_patch_tool_desc, get_edit_tool_desc, get_glob_tool_desc, get_grep_tool_desc,
            get_python_repl_tool_desc, get_read_tool_desc, get_shell_tool_desc,
            get_web_fetch_tool_desc, get_web_search_tool_desc, get_write_tool_desc,
        },
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDesc>,

    /// Sub-agents available to the agent (each registered as a callable tool)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subagents: Vec<AgentSpec>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_options: Option<LangModelOptions>,

    /// Public self-introduction exposed to a calling agent or orchestrator.
    ///
    /// Only relevant when this agent acts as a sub-agent.
    /// `None` for top-level agents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<AgentCard>,

    /// Files this agent pre-fills into the runenv at build time.  Each entry
    /// carries an absolute path and its content, so a serialised `AgentSpec`
    /// reproduces the same runenv layout elsewhere.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileEntry>,

    /// Skill directories owned by this agent.  Each entry is the absolute
    /// path of a directory containing a `SKILL.md` file (plus any supporting
    /// files).  The skill's identifier is the final path segment.  Entries
    /// can live anywhere on the runenv — they do not need to share a parent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<PathBuf>,

    /// Engines used by the `web_search` tool. `None` (or not provided) means
    /// all available engines. Only meaningful when `web_search` is listed in `tools`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_engines: Option<Vec<WebSearchEngineKind>>,
}

impl AgentSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instruction: None,
            tools: Vec::new(),
            subagents: Vec::new(),
            card: None,
            files: Vec::new(),
            skills: Vec::new(),
            model_options: None,
            web_search_engines: None,
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
    /// * `openai/*`: `shell`, `read`, `apply_patch`. Shell-first — `shell` is preferred
    ///   over dedicated `glob`/`grep`, and `apply_patch` is preferred over `write`+`edit`.
    /// * others: `shell`, `read`, `write`, `edit`, `glob`, `grep`.
    pub fn system_tools(mut self) -> Self {
        self.tools.extend(if self.model.starts_with("openai/") {
            vec![
                get_shell_tool_desc(),
                get_read_tool_desc(),
                get_apply_patch_tool_desc(),
            ]
        } else {
            vec![
                get_shell_tool_desc(),
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

    /// Append only the `shell` tool, without the other `system_tools` entries.
    pub fn shell_tool(mut self) -> Self {
        self.tools.push(get_shell_tool_desc());
        self
    }

    /// Add the `web_search` tool to the spec.
    ///
    /// Pass a non-empty `engines` vec to restrict which engines are used;
    /// an empty vec (or `vec![]`) uses all available engines.
    pub fn web_search_tool(mut self, engines: Vec<WebSearchEngineKind>) -> Self {
        self.tools.push(get_web_search_tool_desc());
        if !engines.is_empty() {
            self.web_search_engines = Some(engines);
        }
        self
    }

    /// Add the `web_fetch` tool to the spec.
    ///
    /// Like `web_search_tool`, this is opt-in and is not included in
    /// `system_tools()`. The tool accepts either a single `url` or a `urls`
    /// array (up to five) for parallel fetches, honors robots.txt, and
    /// rate-limits one request per second per host.
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

    pub fn file(mut self, file: FileEntry) -> Self {
        self.files.push(file);
        self
    }

    pub fn files(mut self, files: impl IntoIterator<Item = FileEntry>) -> Self {
        self.files = files.into_iter().collect();
        self
    }

    /// Declare a skill at `dir` together with its pre-fill content.
    /// `dir` is appended to [`Self::skills`]; `entries` are appended to
    /// [`Self::files`].  Every entry's path must be under `dir` — panics
    /// otherwise — so a skill's declared territory and its seeded content
    /// stay aligned.
    pub fn skill(
        mut self,
        dir: impl Into<PathBuf>,
        entries: impl IntoIterator<Item = FileEntry>,
    ) -> Self {
        let dir = dir.into();
        let entries: Vec<FileEntry> = entries.into_iter().collect();
        for e in &entries {
            assert!(
                e.path.starts_with(&dir),
                "skill entry path {:?} must live under skill dir {:?}",
                e.path,
                dir,
            );
        }
        self.skills.push(dir);
        self.files.extend(entries);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_files_omitted_from_serialisation() {
        let spec = AgentSpec::new("openai/gpt-4o-mini");
        let json = serde_json::to_string(&spec).unwrap();
        // Empty files must not surface as "files":[] in the wire form.
        assert!(
            !json.contains("\"files\""),
            "serialised spec should omit empty files: {json}"
        );
    }

    #[test]
    fn test_spec_files_roundtrip() {
        let spec = AgentSpec::new("openai/gpt-4o-mini")
            .instruction("hello")
            .file(FileEntry::new(
                "/workspace/skills/greet/SKILL.md",
                b"---\nname: greet\ndescription: Say hello.\n---\nhi\n".to_vec(),
            ))
            .file(FileEntry::new(
                "/workspace/data/note.txt",
                b"unrelated\n".to_vec(),
            ));
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.files.len(), 2);
        assert_eq!(
            back.files[0].path,
            std::path::PathBuf::from("/workspace/skills/greet/SKILL.md")
        );
        assert_eq!(back.files[1].content.as_ref(), b"unrelated\n");
    }

    #[test]
    fn test_default_skills_empty() {
        let spec = AgentSpec::new("openai/gpt-4o-mini");
        assert!(spec.skills.is_empty());
    }

    #[test]
    fn test_default_skills_omitted_from_serialisation() {
        let spec = AgentSpec::new("openai/gpt-4o-mini");
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            !json.contains("\"skills\""),
            "serialised spec should omit empty skills: {json}"
        );
    }

    #[test]
    #[should_panic(expected = "must live under skill dir")]
    fn test_skill_panics_on_entry_outside_dir() {
        let _ = AgentSpec::new("openai/gpt-4o-mini").skill(
            "/workspace/skills/greet",
            [FileEntry::new(
                "/workspace/elsewhere/SKILL.md",
                b"x".to_vec(),
            )],
        );
    }

    #[test]
    fn test_skill_accepts_entries_under_dir() {
        let spec = AgentSpec::new("openai/gpt-4o-mini").skill(
            "/workspace/skills/greet",
            [
                FileEntry::new("/workspace/skills/greet/SKILL.md", b"a".to_vec()),
                FileEntry::new("/workspace/skills/greet/helper.py", b"b".to_vec()),
            ],
        );
        assert_eq!(spec.skills.len(), 1);
        assert_eq!(spec.files.len(), 2);
    }

    #[test]
    fn test_skills_roundtrip() {
        let spec = AgentSpec::new("openai/gpt-4o-mini")
            .skill("/workspace/skills/greet", [])
            .skill("/workspace/skills/farewell", []);
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.skills.len(), 2);
        assert_eq!(
            back.skills[0],
            std::path::PathBuf::from("/workspace/skills/greet")
        );
        assert_eq!(
            back.skills[1],
            std::path::PathBuf::from("/workspace/skills/farewell")
        );
    }

    #[test]
    fn test_skills_missing_in_payload_falls_back_to_empty() {
        let json = r#"{"model":"openai/gpt-4o-mini"}"#;
        let back: AgentSpec = serde_json::from_str(json).unwrap();
        assert!(back.skills.is_empty());
    }

    #[test]
    fn test_spec_with_subagent_files_roundtrip_recursively() {
        let sub = AgentSpec::new("openai/gpt-4o-mini")
            .card(AgentCard {
                name: "child".into(),
                description: "child agent".into(),
                skills: vec![],
            })
            .file(FileEntry::new(
                "/workspace/skills/child/c/SKILL.md",
                b"---\nname: c\ndescription: child skill\n---\nchild body\n".to_vec(),
            ));
        let parent = AgentSpec::new("openai/gpt-4o-mini")
            .file(FileEntry::new(
                "/workspace/skills/p/SKILL.md",
                b"---\nname: p\ndescription: parent skill\n---\nparent body\n".to_vec(),
            ))
            .subagent(sub);

        let json = serde_json::to_string(&parent).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.files.len(), 1);
        assert_eq!(back.subagents.len(), 1);
        assert_eq!(back.subagents[0].files.len(), 1);
    }

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
