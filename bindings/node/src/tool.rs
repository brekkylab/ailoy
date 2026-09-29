//! A JavaScript function as a [`ToolFunc`].
//!
//! The function is called with the model's arguments as one object, whose fields the tool's
//! parameters schema names. Its return value becomes the tool's result, so it must be what a
//! message can hold: `null`, a boolean, a number, a string, or an array or object of those.
//!
//! # Sync and async
//!
//! A plain return value is the result; a returned promise is awaited. The call goes through a
//! threadsafe function because a JavaScript function can only be called on the JavaScript
//! thread, which a turn leaves free since it is iterated by promises.
//!
//! The threadsafe function is weak, so a registered tool does not keep the process alive.
//!
//! # Failure
//!
//! A throw or rejection becomes the result `"error: ..."` instead of ending the turn, so the
//! model can try differently.
//!
//! A JavaScript tool is not handed the console; one that runs commands holds its own
//! `ConsoleClient`.

use std::sync::Arc;

use ailoy::{
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

use crate::convert::Json;

/// Called with the arguments (not error-first), and weak.
pub type Callback = ThreadsafeFunction<Json<Value>, Returned, Json<Value>, Status, false, true>;

/// What calling a tool's function returned: the result, or a promise of it.
pub enum Returned {
    Ready(Value),
    Pending(Promise<Json<Value>>),
}

impl FromNapiValue for Returned {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let mut is_promise = false;
        napi::check_status!(unsafe { sys::napi_is_promise(env, napi_val, &mut is_promise) })?;
        Ok(if is_promise {
            Returned::Pending(unsafe { Promise::from_napi_value(env, napi_val)? })
        } else {
            Returned::Ready(unsafe { Json::<Value>::from_napi_value(env, napi_val)? }.0)
        })
    }
}

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
                depth: None,
                source_agent: None,
            }
        })
        .boxed()
    })
}

async fn call(func: &Callback, args: Value) -> Result<Value, String> {
    match func.call_async_catch(Json(args)).await.map_err(reason)? {
        Returned::Ready(value) => Ok(value),
        Returned::Pending(promise) => promise.await.map(|json| json.0).map_err(reason),
    }
}

/// The error's own message. A rejection comes back wrapped once more, as
/// `"GenericFailure, <message>"`, which is napi's bookkeeping rather than the tool's.
fn reason(error: napi::Error) -> String {
    let reason = error.reason.as_str();
    reason
        .strip_prefix("GenericFailure, ")
        .unwrap_or(reason)
        .to_string()
}
