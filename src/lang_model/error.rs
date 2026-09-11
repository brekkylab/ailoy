//! The error a model request ends in, with what a caller can act on.

use thiserror::Error;

/// A request to the model API that did not produce a response.
///
/// `status` is the HTTP status when a response arrived, `None` for a transport failure.
/// `retryable` says whether the same request may succeed later — 429/408/5xx/transport —
/// which is what a UI uses to offer "retry" and what the retry loop already acted on
/// (`attempts` is how many times it tried). `message` is the provider's body, verbatim.
#[derive(Debug, Clone, Error)]
#[error("model request failed{}: {message}", status.map(|s| format!(" (HTTP {s})")).unwrap_or_default())]
pub struct ModelError {
    pub status: Option<u16>,
    pub retryable: bool,
    pub message: String,
    pub attempts: u32,
}
