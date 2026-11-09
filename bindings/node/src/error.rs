//! How ailoy's failures arrive in JavaScript.
//!
//! An `Error` with `code` `AILOY_ERROR` for everything ailoy itself reports (an unserved model,
//! an unregistered tool, an API error), and `VIRTX_ERROR` for a sandbox's failure.

use std::future::Future;

use napi::{
    Env, JsError,
    bindgen_prelude::{PromiseRaw, ToNapiValue},
};

pub type Result<T> = napi::Result<T, String>;

pub fn ailoy(reason: impl ToString) -> napi::Error<String> {
    napi::Error::new("AILOY_ERROR".to_string(), reason)
}

pub fn anyhow(error: anyhow::Error) -> napi::Error<String> {
    match error.downcast::<virtx::Error>() {
        Ok(error) => virtx(error),
        Err(error) => ailoy(format!("{error:#}")),
    }
}

/// A sandbox's failure, `VIRTX_ERROR`, its message carrying the server's status where it gave one.
pub fn virtx(error: virtx::Error) -> napi::Error<String> {
    napi::Error::new("VIRTX_ERROR".to_string(), error.to_string())
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
