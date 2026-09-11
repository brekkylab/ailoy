//! How a controlled run ends when it does not end with a message.

use thiserror::Error;

use crate::lang_model::ModelError;

/// The reasons a run stops short. `Cancelled` and `MaxTurns` leave the history
/// consistent (every tool call answered, a partial answer committed); the rest report
/// the failing layer so a caller can decide whether retrying makes sense.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("run cancelled")]
    Cancelled,
    #[error("turn limit reached after {turns} model calls")]
    MaxTurns { turns: u32 },
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("tool execution failed: {0}")]
    Tool(#[source] anyhow::Error),
    #[error("console unavailable: {0}")]
    Console(#[source] anyhow::Error),
    #[error(transparent)]
    Other(anyhow::Error),
}

impl AgentError {
    /// Classify an `anyhow` error from the model layer: a [`ModelError`] inside becomes
    /// [`AgentError::Model`]; anything else is [`AgentError::Other`].
    pub fn from_anyhow(e: anyhow::Error) -> Self {
        match e.downcast::<ModelError>() {
            Ok(m) => AgentError::Model(m),
            Err(e) => AgentError::Other(e),
        }
    }
}
