//! How ailoy's failures arrive in JavaScript.
//!
//! An `Error` with `code` `AILOY_ERROR` for everything ailoy itself reports (an unserved model,
//! an unregistered tool, an API error). Console failures keep cortex's codes, so a refusal
//! inside a turn is the same `TIMED_OUT`, say, as one from `ConsoleClient.exec`.

use cortex::protocol::Failure;

pub use cortex_node::error::{Result, invalid, unsigned};

pub fn ailoy(reason: impl ToString) -> napi::Error<String> {
    napi::Error::new("AILOY_ERROR".to_string(), reason)
}

pub fn anyhow(error: anyhow::Error) -> napi::Error<String> {
    match error.downcast::<Failure>() {
        Ok(failure) => cortex_node::error::failure(failure),
        Err(error) => ailoy(format!("{error:#}")),
    }
}
