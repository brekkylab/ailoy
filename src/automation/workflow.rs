//! `workflow.json`: the tasks of an automation and how they connect.
//!
//! A workflow is a list of tasks. Each task names, under `inputs`, which earlier
//! tasks (or the triggering event) it reads, which makes the list a DAG; the runner
//! executes it one task at a time in topological order, ties broken by list order.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agent::AgentSpec,
    lang_model::ResponseFormat,
    tool::{
        ToolDesc,
        r#impl::{
            get_apply_patch_tool_desc, get_docread_tool_desc, get_edit_tool_desc,
            get_glob_tool_desc, get_grep_tool_desc, get_read_tool_desc, get_shell_tool_desc,
            get_web_fetch_tool_desc, get_web_search_tool_desc, get_write_tool_desc,
        },
    },
};

/// The reserved input source naming the triggering event's payload.
pub const EVENT_INPUT: &str = "event";

/// Built-in tool names an agent task may list.
pub const BUILTIN_TOOLS: [&str; 10] = [
    "shell",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "apply_patch",
    "docread",
    "web_fetch",
    "web_search",
];

/// The [`ToolDesc`] behind a built-in tool name.
pub fn builtin_tool_desc(name: &str) -> anyhow::Result<ToolDesc> {
    Ok(match name {
        "shell" => get_shell_tool_desc(),
        "read" => get_read_tool_desc(),
        "write" => get_write_tool_desc(),
        "edit" => get_edit_tool_desc(),
        "glob" => get_glob_tool_desc(),
        "grep" => get_grep_tool_desc(),
        "apply_patch" => get_apply_patch_tool_desc(),
        "docread" => get_docread_tool_desc(),
        "web_fetch" => get_web_fetch_tool_desc(),
        "web_search" => get_web_search_tool_desc(),
        other => anyhow::bail!(
            "unknown tool `{other}`; built-in tools are {}",
            BUILTIN_TOOLS.join(", ")
        ),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Workflow {
    /// Bound on the whole run, in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Id of the task whose output is the workflow's final result. Defaults to the
    /// last task in the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,

    /// Executed in topological order of `inputs`, ties broken by this order.
    pub tasks: Vec<Task>,
}

fn default_timeout_secs() -> u64 {
    600
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Task {
    /// Unique within the workflow. Also names `tasks/<name>.py` or `tasks/<name>.md`.
    #[schemars(regex(pattern = r"^[A-Za-z_][A-Za-z0-9_-]*$"))]
    pub name: String,

    /// Input name → name of an earlier task, or `event` for the triggering payload.
    /// The task receives a JSON object with these names as keys.
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,

    /// Bound on this task, in seconds. Unset means the run's remaining time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,

    #[serde(flatten)]
    pub kind: TaskKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskKind {
    /// `python3 main.py <input.json> <output.json>` in the console.
    Python {
        /// Inline source. When absent, `tasks/<name>.py` in the automation directory.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,

        /// pypi packages, each pinned as `name==version`. Baked into the console image.
        #[serde(default)]
        packages: Vec<String>,
    },

    /// An agent turn on the shared console, given the rendered prompt.
    Agent {
        /// The agent. Its `tools` must be empty (use the task's `tools`) and it may
        /// not declare `subagents`.
        agent: Box<AgentSpec>,

        /// Prompt template; `{{ name.path }}` reads the task's inputs. When absent,
        /// `tasks/<name>.md` in the automation directory.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,

        /// Built-in tool names the agent gets. Nothing else enters the session.
        #[serde(default)]
        tools: Vec<String>,

        /// JSON Schema the agent's final answer must satisfy. The task's output is then
        /// that JSON rather than the answer's text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_schema: Option<Value>,
    },
}

impl Task {
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            TaskKind::Python { .. } => "python",
            TaskKind::Agent { .. } => "agent",
        }
    }
}

impl Workflow {
    pub fn schema() -> Value {
        serde_json::to_value(schemars::schema_for!(Workflow)).expect("schema serializes")
    }

    /// Everything about the graph that can be wrong without looking at the file
    /// system. Every problem is reported, each prefixed with the task it is about.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut problems = Vec::new();
        if self.tasks.is_empty() {
            problems.push("workflow has no tasks".to_string());
        }

        let mut names = BTreeSet::new();
        for t in &self.tasks {
            if t.name == EVENT_INPUT {
                problems.push(format!(
                    "{}: `{EVENT_INPUT}` is reserved for the trigger payload",
                    t.name
                ));
            }
            if !names.insert(t.name.as_str()) {
                problems.push(format!("{}: duplicate task name", t.name));
            }
        }

        for t in &self.tasks {
            for (name, source) in &t.inputs {
                if source == &t.name {
                    problems.push(format!(
                        "{}: input `{name}` refers to the task itself",
                        t.name
                    ));
                } else if source != EVENT_INPUT && !names.contains(source.as_str()) {
                    problems.push(format!(
                        "{}: input `{name}` refers to unknown task `{source}`",
                        t.name
                    ));
                }
            }
            match &t.kind {
                TaskKind::Python { packages, .. } => {
                    for p in packages {
                        if !is_pinned(p) {
                            problems.push(format!(
                                "{}: package `{p}` must be pinned as name==version",
                                t.name
                            ));
                        }
                    }
                }
                TaskKind::Agent {
                    agent,
                    tools,
                    output_schema,
                    ..
                } => {
                    if !agent.tools.is_empty() {
                        problems.push(format!(
                            "{}: `agent.tools` must be empty; list tool names in the task's `tools`",
                            t.name
                        ));
                    }
                    if !agent.subagents.is_empty() {
                        problems.push(format!("{}: `agent.subagents` is not allowed", t.name));
                    }
                    for name in tools {
                        if let Err(e) = builtin_tool_desc(name) {
                            problems.push(format!("{}: {e}", t.name));
                        }
                    }
                    if let Some(schema) = output_schema
                        && let Err(e) = ResponseFormat::json_schema(schema.clone().into())
                    {
                        problems.push(format!("{}: output_schema: {e}", t.name));
                    }
                }
            }
        }

        if let Some(out) = &self.output
            && !names.contains(out.as_str())
        {
            problems.push(format!("output: unknown task `{out}`"));
        }

        // One version per package name across the workflow: they share one image.
        let mut versions: HashMap<String, &str> = HashMap::new();
        for p in self.all_packages() {
            if let Some((name, version)) = p.split_once("==") {
                let key = normalize_package_name(name);
                match versions.get(&key) {
                    Some(v) if *v != version => problems.push(format!(
                        "package `{name}` is pinned to both {v} and {version}"
                    )),
                    Some(_) => {}
                    None => {
                        versions.insert(key, version);
                    }
                }
            }
        }

        if problems.is_empty()
            && let Err(e) = self.order()
        {
            problems.push(e.to_string());
        }

        if problems.is_empty() {
            Ok(())
        } else {
            anyhow::bail!("{}", problems.join("\n"))
        }
    }

    /// Execution order: Kahn's algorithm, always taking the first ready task in list
    /// order. Fails on a cycle, naming the tasks left in it.
    pub fn order(&self) -> anyhow::Result<Vec<&Task>> {
        let mut done: BTreeSet<&str> = BTreeSet::new();
        let mut remaining: Vec<&Task> = self.tasks.iter().collect();
        let mut order = Vec::with_capacity(self.tasks.len());
        while !remaining.is_empty() {
            let ready = remaining.iter().position(|t| {
                t.inputs
                    .values()
                    .all(|src| src == EVENT_INPUT || done.contains(src.as_str()))
            });
            match ready {
                Some(i) => {
                    let t = remaining.remove(i);
                    done.insert(&t.name);
                    order.push(t);
                }
                None => anyhow::bail!(
                    "cycle among tasks: {}",
                    remaining
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
        Ok(order)
    }

    pub fn task(&self, name: &str) -> Option<&Task> {
        self.tasks.iter().find(|t| t.name == name)
    }

    /// The task whose output is the run's result.
    pub fn output_task(&self) -> Option<&Task> {
        match &self.output {
            Some(name) => self.task(name),
            None => self.tasks.last(),
        }
    }

    /// Every task's packages, in declaration order, duplicates included.
    fn all_packages(&self) -> impl Iterator<Item = &str> {
        self.tasks.iter().flat_map(|t| match &t.kind {
            TaskKind::Python { packages, .. } => packages.iter().map(String::as_str).collect(),
            TaskKind::Agent { .. } => Vec::new(),
        })
    }

    /// Sorted, deduplicated packages of the whole workflow: what the console image
    /// carries.
    pub fn packages(&self) -> Vec<String> {
        let set: BTreeSet<&str> = self.all_packages().collect();
        set.into_iter().map(str::to_string).collect()
    }
}

/// `name==version`, with a PEP 508 name and a non-empty version.
fn is_pinned(spec: &str) -> bool {
    let Some((name, version)) = spec.split_once("==") else {
        return false;
    };
    let name_ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && name.starts_with(|c: char| c.is_ascii_alphanumeric());
    let version_ok = !version.is_empty()
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '!' | '-'));
    name_ok && version_ok
}

/// PEP 503: case-insensitive, `-`/`_`/`.` runs are one `-`.
fn normalize_package_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_sep = false;
    for c in name.chars() {
        if matches!(c, '-' | '_' | '.') {
            if !last_sep {
                out.push('-');
            }
            last_sep = true;
        } else {
            out.push(c.to_ascii_lowercase());
            last_sep = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn wf(v: Value) -> Workflow {
        serde_json::from_value(v).unwrap()
    }

    fn py(name: &str, inputs: Value) -> Value {
        json!({"name": name, "kind": "python", "inputs": inputs})
    }

    #[test]
    fn parses_with_defaults() {
        let w = wf(json!({"tasks": [py("a", json!({}))]}));
        assert_eq!(w.timeout_secs, 600);
        assert!(w.output.is_none());
        assert_eq!(w.output_task().unwrap().name, "a");
        w.validate().unwrap();
    }

    #[test]
    fn order_is_topological_with_list_tiebreak() {
        let w = wf(json!({"tasks": [
            py("c", json!({"x": "a", "y": "b"})),
            py("b", json!({"x": "a"})),
            py("a", json!({"e": "event"})),
            py("d", json!({})),
        ]}));
        let names: Vec<_> = w.order().unwrap().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c", "d"]);
    }

    #[test]
    fn rejects_cycles_unknown_refs_and_reserved_id() {
        let err = wf(json!({"tasks": [
            py("a", json!({"x": "b"})),
            py("b", json!({"x": "a"})),
        ]}))
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("cycle"), "{err}");

        let err = wf(json!({"tasks": [py("a", json!({"x": "nope"}))]}))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown task `nope`"), "{err}");

        let err = wf(json!({"tasks": [py("event", json!({}))]}))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("reserved"), "{err}");

        let err = wf(json!({"tasks": [py("a", json!({})), py("a", json!({}))]}))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"), "{err}");

        let err = wf(json!({"output": "zzz", "tasks": [py("a", json!({}))]}))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("output: unknown task"), "{err}");
    }

    #[test]
    fn rejects_unpinned_and_conflicting_packages() {
        let err = wf(json!({"tasks": [
            {"name": "a", "kind": "python", "packages": ["requests"]},
        ]}))
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("pinned"), "{err}");

        let err = wf(json!({"tasks": [
            {"name": "a", "kind": "python", "packages": ["Requests==1.0"]},
            {"name": "b", "kind": "python", "packages": ["requests==2.0"]},
        ]}))
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("both 1.0 and 2.0"), "{err}");

        let w = wf(json!({"tasks": [
            {"name": "a", "kind": "python", "packages": ["b==2", "a==1"]},
            {"name": "b", "kind": "python", "packages": ["a==1"]},
        ]}));
        w.validate().unwrap();
        assert_eq!(w.packages(), ["a==1", "b==2"]);
    }

    #[test]
    fn rejects_agent_spec_tools_subagents_and_unknown_tool_names() {
        let base = |agent: Value, tools: Value| {
            wf(
                json!({"tasks": [{"name": "r", "kind": "agent", "agent": agent, "tools": tools,
                                 "prompt": "hi"}]}),
            )
        };
        let err = base(
            json!({"model": "m", "tools": [{"name": "x", "parameters": {}}]}),
            json!([]),
        )
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("`agent.tools` must be empty"), "{err}");

        let err = base(
            json!({"model": "m", "subagents": [{"model": "m"}]}),
            json!([]),
        )
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("subagents"), "{err}");

        let err = base(json!({"model": "m"}), json!(["shell", "teleport"]))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("teleport"), "{err}");

        base(json!({"model": "m"}), json!(["shell", "read"]))
            .validate()
            .unwrap();
    }

    #[test]
    fn rejects_invalid_output_schema() {
        let err = wf(
            json!({"tasks": [{"name": "r", "kind": "agent", "agent": {"model": "m"},
                                         "prompt": "hi", "output_schema": {"type": 123}}]}),
        )
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("output_schema"), "{err}");
    }

    #[test]
    fn schema_tags_task_kinds() {
        let s = Workflow::schema().to_string();
        assert!(s.contains("\"python\"") && s.contains("\"agent\""), "{s}");
    }
}
