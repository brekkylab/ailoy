//! A JavaScript function as a [`ToolFunc`].
//!
//! The model's arguments arrive as the one argument the function is called with — an object,
//! as the parameters schema a tool is described by names its fields. What the function returns
//! becomes the tool's result value, so it has to be what a message can hold: `null`, a
//! boolean, a number, a string, or an array or object of those.
//!
//! # Sync and async
//!
//! Either is taken, and which is told by what calling it returns. A plain result is the
//! answer; a promise is awaited. The call itself is made on the JavaScript thread, through a
//! threadsafe function, since that is the only thread a function can be called on — which a
//! turn leaves free, because it is iterated by promises.
//!
//! The threadsafe function is weak: a registered tool does not keep the process alive by
//! itself, as a function held in a registry should not.
//!
//! # Failure
//!
//! A tool that throws, or whose promise rejects, answers with the error as its result,
//! `"error: ..."`, rather than ending the turn: the model asked for something that did not
//! work, and it is the one that can try differently.
//!
//! A JavaScript tool is pure — it is not handed the console. One that needs to run a command
//! does it through a `Console` it holds itself.

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

/// What a tool's function is taken as: called with the arguments, not an error first, and
/// weak.
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
