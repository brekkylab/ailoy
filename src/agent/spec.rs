use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentCard,
    runenv::FileEntry,
    tool::{
        ToolDesc,
        r#impl::{
            get_apply_patch_tool_desc, get_bash_tool_desc, get_edit_tool_desc, get_glob_tool_desc,
            get_grep_tool_desc, get_python_repl_tool_desc, get_read_tool_desc,
            get_web_search_tool_desc, get_write_tool_desc,
        },
    },
};

/// Default location inside the runenv where this agent's skill files live.
/// Sub-agent skill dirs are derived by nesting under the parent's `skill_dir`
/// at build time (see [`Agent::try_with_provider_and_runenv`]).
pub fn default_skill_dir() -> PathBuf {
    PathBuf::from("/workspace/skills")
}

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

    /// Directory inside the runenv where this agent's skill files live.
    /// Files in [`AgentSpec::files`] whose path matches
    /// `<skill_dir>/<name>/SKILL.md` are surfaced as skills.  Sub-agent skill
    /// dirs are re-rooted under the parent's `skill_dir` at build time, so a
    /// sub-spec's declared value is overwritten with the nested layout.
    #[serde(default = "default_skill_dir")]
    pub skill_dir: PathBuf,
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
            skill_dir: default_skill_dir(),
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

    pub fn file(mut self, file: FileEntry) -> Self {
        self.files.push(file);
        self
    }

    pub fn files(mut self, files: impl IntoIterator<Item = FileEntry>) -> Self {
        self.files = files.into_iter().collect();
        self
    }

    /// Override the directory inside the runenv where this agent's skill
    /// files live (defaults to `/workspace/skills`).  When this spec is used
    /// as a sub-agent, the parent's build step overwrites this value with the
    /// nested layout `<parent.skill_dir>/<card.name>`.
    pub fn skill_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skill_dir = dir.into();
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
    fn test_default_skill_dir() {
        let spec = AgentSpec::new("openai/gpt-4o-mini");
        assert_eq!(spec.skill_dir, std::path::PathBuf::from("/workspace/skills"));
    }

    #[test]
    fn test_skill_dir_roundtrip() {
        let spec = AgentSpec::new("openai/gpt-4o-mini").skill_dir("/tmp/custom/skills");
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.skill_dir, std::path::PathBuf::from("/tmp/custom/skills"));
    }

    #[test]
    fn test_skill_dir_missing_in_payload_falls_back_to_default() {
        // Older spec payloads written before `skill_dir` existed should still
        // deserialise — `#[serde(default)]` fills in the canonical location.
        let json = r#"{"model":"openai/gpt-4o-mini"}"#;
        let back: AgentSpec = serde_json::from_str(json).unwrap();
        assert_eq!(back.skill_dir, std::path::PathBuf::from("/workspace/skills"));
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
}
