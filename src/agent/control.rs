//! What a caller can do to a run while it runs: stop it, bound it, and vet its tool calls.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::datatype::Value;

/// One tool call the model asked for, before it runs.
#[derive(Debug)]
pub struct ToolCallRequest<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a Value,
}

/// A gate's answer. `Deny` becomes the tool's result — the model reads the reason and
/// continues — rather than an error, so a refusal never wedges the history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolDecision {
    Allow,
    Deny { reason: String },
}

/// Reviews each tool call between the model asking and the runtime executing. An
/// implementation may await a person; the run waits with it.
#[async_trait]
pub trait ToolGate: Send + Sync {
    async fn review(&self, call: ToolCallRequest<'_>) -> ToolDecision;
}

/// The default gate: everything runs.
pub struct AllowAll;

#[async_trait]
impl ToolGate for AllowAll {
    async fn review(&self, _call: ToolCallRequest<'_>) -> ToolDecision {
        ToolDecision::Allow
    }
}

/// Controls for one `run_stream_controlled` call.
///
/// `Clone` shares rather than copies: the clone holds the same [`CancellationToken`] and
/// the same `Arc<dyn ToolGate>`, so cancelling either handle cancels every run built from
/// them, and every such run is vetted by the one gate. `max_turns` is the only field a
/// clone owns outright. Hand out a clone to cancel a run from elsewhere; build a fresh
/// `RunControl` when a run needs a cancel of its own.
#[derive(Clone)]
pub struct RunControl {
    /// Cancel at any await point. The runtime commits what it has and answers pending
    /// tool calls with stubs before returning `AgentError::Cancelled`.
    pub cancel: CancellationToken,
    /// Upper bound on model calls in this run. `None` is unbounded (the pre-existing
    /// behaviour of `run_stream`).
    pub max_turns: Option<u32>,
    pub tool_gate: Arc<dyn ToolGate>,
}

impl Default for RunControl {
    fn default() -> Self {
        Self {
            cancel: CancellationToken::new(),
            max_turns: None,
            tool_gate: Arc::new(AllowAll),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allow_all_allows() {
        let gate = AllowAll;
        let args = crate::datatype::Value::object_empty();
        let d = gate
            .review(ToolCallRequest {
                id: "c1",
                name: "shell",
                arguments: &args,
            })
            .await;
        assert!(matches!(d, ToolDecision::Allow));
    }

    #[test]
    fn default_control_is_unbounded_and_open() {
        let ctl = RunControl::default();
        assert!(ctl.max_turns.is_none());
        assert!(!ctl.cancel.is_cancelled());
    }

    #[test]
    fn from_anyhow_classifies_model_errors() {
        let me = crate::lang_model::ModelError {
            status: Some(401),
            retryable: false,
            message: "no".into(),
            attempts: 1,
        };
        let e: anyhow::Error = me.into();
        assert!(
            matches!(crate::agent::AgentError::from_anyhow(e), crate::agent::AgentError::Model(m) if m.status == Some(401))
        );
        assert!(matches!(
            crate::agent::AgentError::from_anyhow(anyhow::anyhow!("x")),
            crate::agent::AgentError::Other(_)
        ));
    }
}
