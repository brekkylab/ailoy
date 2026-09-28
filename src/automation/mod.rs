//! Automations: agent runs that begin with an event rather than with a person.
//!
//! An automation is a directory:
//!
//! ```text
//! <automation>/
//!   console.json    where the tasks run: image, whether there is a network (ConsoleSpec)
//!   workflow.json   what runs: tasks and how they connect                (Workflow)
//!   tasks/<name>.py a python task's code, when not written inline
//!   tasks/<name>.md an agent task's prompt template, when not written inline
//! ```
//!
//! What starts a run is not in the tree. A trigger — which sources to watch and what
//! to make of them — belongs to whoever registers the automation with a daemon; this
//! module only knows the [`Event`] it hands over. The daemon's own draft of that
//! registration may sit beside these files as `trigger.json`, unread here.
//!
//! A workflow is a DAG of tasks, run one at a time in topological order on **one
//! console**. A `python` task is `python3 main.py input.json output.json`; an `agent`
//! task is one agent turn given a prompt rendered from its inputs. What flows between
//! tasks is a JSON value; what an agent sees is exactly the rendered prompt. The
//! console's host-mounted trees (scratch, artifacts) and its working directory carry
//! over from task to task, while the guest itself is torn down and booted again around
//! every batch of tool calls — so pip packages are baked into the image up front.
//!
//! A [`Runner`] is `(definition, event) → record`. It lays out `runs/<id>/`, opens the
//! console, runs the tasks and writes a [`RunRecord`]. Safety is not in checking what
//! the tasks produce: it is in the console contract the server enforces and in the
//! record kept for audit.

mod authoring;
mod console_spec;
mod definition;
mod event;
mod record;
mod render;
mod runner;
mod workflow;

pub use authoring::{
    CONSOLE_SCHEMA_FILE, SKILL, SKILL_FILE, WORKFLOW_SCHEMA_FILE, authoring_files,
};
pub use console_spec::{ConsoleSpec, GUEST_ARTIFACTS, GUEST_SCRATCH};
pub use definition::{AutomationDef, CONSOLE_FILE, TASKS_DIR, WORKFLOW_FILE};
pub use event::Event;
pub use record::{Outcome, RunRecord, TaskDetail, TaskRecord};
pub use render::render;
pub use runner::{PreparedRun, RECORD_FILE, RUNS_DIR, Runner, TASKS_SCRATCH_DIR};
pub use workflow::{BUILTIN_TOOLS, EVENT_INPUT, Task, TaskKind, Workflow, builtin_tool_desc};
