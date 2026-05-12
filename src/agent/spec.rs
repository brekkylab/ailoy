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

    /// Skill directories owned by this agent.  Each entry is the absolute
    /// path of a directory containing a `SKILL.md` file (plus any supporting
    /// files).  The skill's identifier is the final path segment.  Entries
    /// can live anywhere on the runenv — they do not need to share a parent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<PathBuf>,

    /// Single fixed directory where the agent creates new skills at runtime.
    /// At [`Agent::snapshot`](crate::agent::Agent::snapshot) time this dir
    /// is scanned once for `<child>/SKILL.md` entries not already in
    /// [`Self::skills`]; new ones are merged into the returned spec.  When
    /// `None`, snapshot does no auto-discovery — only the declared list
    /// round-trips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_root: Option<PathBuf>,
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
            skill_root: None,
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

    /// Declare a skill by the absolute path of the directory containing its
    /// `SKILL.md`.  The directory's last path segment becomes the skill name.
    pub fn skill(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skills.push(dir.into());
        self
    }

    /// Declare multiple skills at once — same semantics as [`Self::skill`].
    pub fn skills(mut self, dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        self.skills = dirs.into_iter().collect();
        self
    }

    /// Set the fixed directory where the agent creates new skills at
    /// runtime.  At snapshot time this dir is scanned once for new
    /// `<child>/SKILL.md` entries; matches not already in [`Self::skills`]
    /// are added to the round-tripped spec.
    pub fn skill_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.skill_root = Some(root.into());
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
    fn test_skills_roundtrip() {
        let spec = AgentSpec::new("openai/gpt-4o-mini")
            .skill("/workspace/skills/greet")
            .skill("/workspace/skills/farewell");
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.skills.len(), 2);
        assert_eq!(back.skills[0], std::path::PathBuf::from("/workspace/skills/greet"));
        assert_eq!(back.skills[1], std::path::PathBuf::from("/workspace/skills/farewell"));
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
}
