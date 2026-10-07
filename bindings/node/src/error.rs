//! How ailoy's failures arrive in JavaScript.
//!
//! An `Error` with `code` `AILOY_ERROR` for everything ailoy itself reports (an unserved model,
//! an unregistered tool, an API error). Console failures keep the codes `@brekkylab/virtx` gives
//! them, so a refusal inside a turn is the same `TIMED_OUT`, say, as one from
//! `ConsoleClient.exec`.

use std::future::Future;

use napi::{
    Env, JsError,
    bindgen_prelude::{PromiseRaw, ToNapiValue},
};
use virtx::protocol::{Error, Failure};

pub type Result<T> = napi::Result<T, String>;

pub fn ailoy(reason: impl ToString) -> napi::Error<String> {
    napi::Error::new("AILOY_ERROR".to_string(), reason)
}

pub fn anyhow(error: anyhow::Error) -> napi::Error<String> {
    match error.downcast::<Failure>() {
        Ok(failure) => self::failure(failure),
        Err(error) => ailoy(format!("{error:#}")),
    }
}

/// As `@brekkylab/virtx` codes it: a refusal by virtx's name for its number (`TIMED_OUT`,
/// `NOT_FOUND`), or `CONSOLE_REFUSED` with the number in the message for a number with no name
/// here; a broken channel `CONSOLE_BROKEN`.
fn failure(failure: Failure) -> napi::Error<String> {
    match failure {
        Failure::Refused(error) => match name(error.code) {
            Some(name) => napi::Error::new(name.to_string(), error.message),
            None => napi::Error::new(
                "CONSOLE_REFUSED".to_string(),
                format!("{} ({})", error.message, error.code),
            ),
        },
        Failure::Broken(error) => {
            napi::Error::new("CONSOLE_BROKEN".to_string(), format!("{error:#}"))
        }
    }
}

fn name(code: i64) -> Option<&'static str> {
    Some(match code {
        Error::TIMED_OUT => "TIMED_OUT",
        Error::NOT_EXECUTABLE => "NOT_EXECUTABLE",
        Error::BOOT_FAILED => "BOOT_FAILED",
        Error::NOT_FOUND => "NOT_FOUND",
        Error::IS_A_DIRECTORY => "IS_A_DIRECTORY",
        Error::IO_FAILED => "IO_FAILED",
        Error::UNSUPPORTED_MOUNT => "UNSUPPORTED_MOUNT",
        Error::MOUNT_FAILED => "MOUNT_FAILED",
        Error::UNSUPPORTED_NETWORK => "UNSUPPORTED_NETWORK",
        Error::UNSUPPORTED_IMAGE => "UNSUPPORTED_IMAGE",
        Error::UNKNOWN_IMAGE => "UNKNOWN_IMAGE",
        Error::UNSUPPORTED_MACHINE => "UNSUPPORTED_MACHINE",
        Error::INVALID_REQUEST => "INVALID_REQUEST",
        Error::METHOD_NOT_FOUND => "METHOD_NOT_FOUND",
        Error::INVALID_PARAMS => "INVALID_PARAMS",
        Error::INTERNAL_ERROR => "INTERNAL_ERROR",
        _ => return None,
    })
}

pub fn invalid(reason: impl ToString) -> napi::Error<String> {
    napi::Error::new("INVALID_ARG".to_string(), reason)
}

/// A JavaScript number where ailoy takes a `u64` (a token count, `topK`).
///
/// napi converts numbers to `i64`; a negative one is rejected here rather than wrapped into a
/// huge count.
pub fn unsigned(value: Option<i64>, what: &str) -> Result<Option<u64>> {
    value
        .map(|v| u64::try_from(v).map_err(|_| invalid(format!("{what} must not be negative"))))
        .transpose()
}

/// Run `fut` on napi's runtime, rejecting with its error's own `code`.
///
/// Not napi's `async fn` or `spawn_future`, which reject with a [`Status`](napi::Status) as the
/// `code`: this runs `fut` to a plain `Result` and builds the JavaScript error back on the main
/// thread, where one can be made.
pub fn promise<'env, T, F>(env: &'env Env, fut: F) -> napi::Result<PromiseRaw<'env, T>>
where
    T: ToNapiValue + Send + 'static,
    F: Future<Output = Result<T>> + Send + 'static,
{
    env.spawn_future_with_callback(async move { Ok(fut.await) }, |env, result| {
        result.map_err(|e| napi::Error::from(JsError::from(e).into_unknown(*env)))
    })
}

/// Our error as a thrown JavaScript error, for paths that must return napi's own error type.
pub fn thrown(env: &Env, error: napi::Error<String>) -> napi::Error {
    napi::Error::from(JsError::from(error).into_unknown(*env))
}
