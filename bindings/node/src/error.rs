//! How ailoy's failures arrive in JavaScript.
//!
//! As an `Error` whose `code` says what kind, as cortex's binding does. `AILOY_ERROR` for
//! everything ailoy itself reports — a model that no provider serves, a tool that is not
//! registered, an API that answered with an error. What a console reports keeps cortex's own
//! codes: a refusal inside an agent's turn is the same `TIMED_OUT`, say, as one from
//! `Console.exec`, so a caller handles it once.

use cortex::console::Failure;

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
