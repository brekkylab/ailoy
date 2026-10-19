//! A JavaScript function as a [`ToolFunc`].
//!
//! The function is called with the model's arguments as one object, whose fields the tool's
//! parameters schema names, and returns the result or a promise of it. The result must be what
//! a message can hold: `null`, a boolean, a number, a string, or an array or object of those.
//!
//! A JavaScript tool is not handed the console; one that runs commands holds its own
//! `ConsoleClient`.

use std::sync::Arc;

use ailoy::{
    browser::AskResult,
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    tool::ToolFunc,
};
use futures::StreamExt as _;
use napi::{
    Status,
    bindgen_prelude::{FromNapiValue, Promise},
    sys,
    threadsafe_function::ThreadsafeFunction,
};

use serde::de::DeserializeOwned;

use crate::convert::Json;

/// Called with the arguments (not error-first). Threadsafe because a JavaScript function runs
/// only on the JavaScript thread, which a turn leaves free as promises iterate it; weak so a
/// registered tool does not keep the process alive.
pub type Callback =
    ThreadsafeFunction<Json<Value>, Returned<Value>, Json<Value>, Status, false, true>;

/// What calling a JavaScript function returned: the result, or a promise of it.
pub enum Returned<T: DeserializeOwned + Send + 'static> {
    Ready(T),
    Pending(Promise<Json<T>>),
}

impl<T: DeserializeOwned + Send + 'static> Returned<T> {
    /// The result, once a promise of it settles.
    pub async fn settle(self) -> Result<T, String> {
        match self {
            Returned::Ready(value) => Ok(value),
            Returned::Pending(promise) => promise.await.map(|json| json.0).map_err(reason),
        }
    }
}

impl<T: DeserializeOwned + Send + 'static> FromNapiValue for Returned<T> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let mut is_promise = false;
        napi::check_status!(unsafe { sys::napi_is_promise(env, napi_val, &mut is_promise) })?;
        Ok(if is_promise {
            Returned::Pending(unsafe { Promise::from_napi_value(env, napi_val)? })
        } else {
            Returned::Ready(unsafe { Json::<T>::from_napi_value(env, napi_val)? }.0)
        })
    }
}

/// A throw or rejection becomes the result `"error: ..."` instead of ending the turn, so the
/// model can try differently.
pub fn tool_func(func: Callback) -> ToolFunc {
    let func = Arc::new(func);
    ToolFunc::new(move |args, id| {
        let func = func.clone();
        futures::stream::once(async move {
            let value = call(&func, args)
                .await
                .unwrap_or_else(|e| Value::string(format!("error: {e}")));
            MessageOutput {
                message: Message::new(Role::Tool)
                    .with_contents([Part::value(value)])
                    .with_id(id),
                finish_reason: FinishReason::Stop {},
                usage: None,
            }
        })
        .boxed()
    })
}

/// A JavaScript function as an `onAsk` handler: called with the request, it returns (or resolves
/// to) `'declined'`, `'done'` or nothing, or any other string, which is done with a note for the
/// agent. A throw is a decline, so the agent does not go on as if the person had answered.
pub async fn ask(func: Arc<Callback>, message: String) -> AskResult {
    match call(&func, Value::string(message)).await {
        Ok(answer) => match answer.as_str() {
            Some("declined") => AskResult::Declined,
            Some("done") | None => AskResult::Done,
            Some(note) => AskResult::Note(note.to_string()),
        },
        Err(_) => AskResult::Declined,
    }
}

async fn call(func: &Callback, args: Value) -> Result<Value, String> {
    func.call_async_catch(Json(args))
        .await
        .map_err(reason)?
        .settle()
        .await
}

/// The error's own message. A rejection comes back wrapped once more, as
/// `"GenericFailure, <message>"`, which is napi's bookkeeping rather than the tool's.
pub fn reason(error: napi::Error) -> String {
    let reason = error.reason.as_str();
    reason
        .strip_prefix("GenericFailure, ")
        .unwrap_or(reason)
        .to_string()
}
