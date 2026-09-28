//! The automation directory: the files a run is made from.

use std::{fs, path::PathBuf};

use super::{
    ConsoleSpec, Workflow,
    workflow::{Task, TaskKind},
};

pub const CONSOLE_FILE: &str = "console.json";
pub const WORKFLOW_FILE: &str = "workflow.json";
/// Task code and prompts not written inline: `tasks/<name>.py`, `tasks/<name>.md`.
pub const TASKS_DIR: &str = "tasks";

/// One automation as it stands on disk. Only what a run needs is here: the trigger
/// that produced the event lives with whoever registered it, not in this tree.
#[derive(Clone, Debug)]
pub struct AutomationDef {
    pub root: PathBuf,
    pub console: ConsoleSpec,
    pub workflow: Workflow,
}

impl AutomationDef {
    /// Read and validate. Every problem found is reported, so a definition written by
    /// an agent can be corrected in one pass.
    pub fn load(root: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let root = root.into();
        let read = |name: &str| {
            fs::read_to_string(root.join(name))
                .map_err(|e| anyhow::anyhow!("reading {}: {e}", root.join(name).display()))
        };
        let console: ConsoleSpec = serde_json::from_str(&read(CONSOLE_FILE)?)
            .map_err(|e| anyhow::anyhow!("{CONSOLE_FILE}: {e}"))?;
        let workflow: Workflow = serde_json::from_str(&read(WORKFLOW_FILE)?)
            .map_err(|e| anyhow::anyhow!("{WORKFLOW_FILE}: {e}"))?;
        workflow
            .validate()
            .map_err(|e| anyhow::anyhow!("{WORKFLOW_FILE}:\n{e}"))?;

        let def = Self {
            root,
            console,
            workflow,
        };
        let mut problems = Vec::new();
        for t in &def.workflow.tasks {
            let result = match &t.kind {
                TaskKind::Python { .. } => def.task_code(t).map(drop),
                TaskKind::Agent { .. } => def.task_prompt(t).map(drop),
            };
            if let Err(e) = result {
                problems.push(e.to_string());
            }
        }
        if !problems.is_empty() {
            anyhow::bail!("{WORKFLOW_FILE}:\n{}", problems.join("\n"));
        }
        Ok(def)
    }

    /// A python task's source: inline `code`, else `tasks/<name>.py`.
    pub fn task_code(&self, task: &Task) -> anyhow::Result<String> {
        match &task.kind {
            TaskKind::Python {
                code: Some(code), ..
            } => Ok(code.clone()),
            TaskKind::Python { code: None, .. } => self.task_file(task, "py", "code"),
            TaskKind::Agent { .. } => anyhow::bail!("{}: not a python task", task.name),
        }
    }

    /// An agent task's prompt template: inline `prompt`, else `tasks/<name>.md`.
    pub fn task_prompt(&self, task: &Task) -> anyhow::Result<String> {
        match &task.kind {
            TaskKind::Agent {
                prompt: Some(prompt),
                ..
            } => Ok(prompt.clone()),
            TaskKind::Agent { prompt: None, .. } => self.task_file(task, "md", "prompt"),
            TaskKind::Python { .. } => anyhow::bail!("{}: not an agent task", task.name),
        }
    }

    fn task_file(&self, task: &Task, ext: &str, field: &str) -> anyhow::Result<String> {
        let path = self.task_path(task, ext);
        fs::read_to_string(&path).map_err(|e| {
            anyhow::anyhow!(
                "{}: no inline `{field}` and {} cannot be read: {e}",
                task.name,
                path.strip_prefix(&self.root).unwrap_or(&path).display()
            )
        })
    }

    fn task_path(&self, task: &Task, ext: &str) -> PathBuf {
        self.root
            .join(TASKS_DIR)
            .join(format!("{}.{ext}", task.name))
    }
}

/// Write a minimal definition for tests elsewhere in this module.
#[cfg(test)]
pub(super) fn write_def(root: &std::path::Path, workflow: &serde_json::Value) {
    fs::create_dir_all(root.join(TASKS_DIR)).unwrap();
    fs::write(
        root.join(CONSOLE_FILE),
        serde_json::json!({"image": "python:3.12-slim"}).to_string(),
    )
    .unwrap();
    fs::write(root.join(WORKFLOW_FILE), workflow.to_string()).unwrap();
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn loads_inline_and_file_backed_tasks() {
        let d = tempfile::tempdir().unwrap();
        write_def(
            d.path(),
            &json!({"tasks": [
                {"name": "a", "kind": "python", "code": "print(1)"},
                {"name": "b", "kind": "python", "inputs": {"x": "a"}},
                {"name": "c", "kind": "agent", "agent": {"model": "m"}, "inputs": {"x": "b"}},
            ]}),
        );
        fs::write(d.path().join("tasks/b.py"), "print(2)").unwrap();
        fs::write(d.path().join("tasks/c.md"), "Do {{ x }}").unwrap();

        let def = AutomationDef::load(d.path()).unwrap();
        let w = &def.workflow;
        assert_eq!(def.task_code(w.task("a").unwrap()).unwrap(), "print(1)");
        assert_eq!(def.task_code(w.task("b").unwrap()).unwrap(), "print(2)");
        assert_eq!(def.task_prompt(w.task("c").unwrap()).unwrap(), "Do {{ x }}");
    }

    #[test]
    fn missing_task_files_are_all_reported() {
        let d = tempfile::tempdir().unwrap();
        write_def(
            d.path(),
            &json!({"tasks": [
                {"name": "a", "kind": "python"},
                {"name": "b", "kind": "agent", "agent": {"model": "m"}},
            ]}),
        );
        let err = AutomationDef::load(d.path()).unwrap_err().to_string();
        assert!(
            err.contains("a: no inline `code`") && err.contains("tasks/a.py"),
            "{err}"
        );
        assert!(
            err.contains("b: no inline `prompt`") && err.contains("tasks/b.md"),
            "{err}"
        );
    }

    #[test]
    fn graph_problems_fail_the_load() {
        let d = tempfile::tempdir().unwrap();
        write_def(
            d.path(),
            &json!({"tasks": [{"name": "a", "kind": "python", "code": "", "inputs": {"x": "a"}}]}),
        );
        let err = AutomationDef::load(d.path()).unwrap_err().to_string();
        assert!(err.contains("refers to the task itself"), "{err}");
    }
}
