//! What one run leaves behind for audit.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Event;
use crate::message::Message;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    Failed {
        error: String,
    },
    TimedOut,
    /// Not attempted, because an earlier task ended the run.
    Skipped,
}

impl Outcome {
    pub fn is_completed(&self) -> bool {
        matches!(self, Outcome::Completed)
    }
}

/// What a task did, by kind.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskDetail {
    Python {
        code: String,
        exit_code: Option<i32>,
        stdout: String,
        stderr: String,
    },
    Agent {
        /// The prompt as the model saw it.
        prompt: String,
        /// The full history, system message first.
        transcript: Vec<Message>,
    },
    /// A skipped task has nothing to show.
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRecord {
    pub name: String,
    /// Unix seconds.
    pub started_at: u64,
    pub finished_at: u64,
    pub input: Value,
    pub output: Option<Value>,
    pub outcome: Outcome,
    pub detail: TaskDetail,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub event: Event,
    /// Unix seconds.
    pub started_at: u64,
    pub finished_at: u64,
    pub outcome: Outcome,
    /// The workflow's final result: the `output` task's output, when it completed.
    pub output: Option<Value>,
    /// One entry per task, in execution order.
    pub tasks: Vec<TaskRecord>,
}
