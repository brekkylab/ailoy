//! A JavaScript function as a [`ToolFunc`].
//!
//! The function is called with the model's arguments as one object, whose fields the tool's
//! parameters schema names, and returns the result or a promise of it. The result must be what
//! a message can hold: `null`, a boolean, a number, a string, or an array or object of those.
//!
//! An async generator streams its result instead: each value it yields is a piece of it, a
//! string as text and anything else as a value, and its end finishes the result. A turn let go
//! of mid-call calls the generator's `return()`, so its `finally` runs once the step under way
//! settles.
//!
//! A JavaScript tool is not handed the console; one that runs commands holds its own
//! `ConsoleClient`.

use std::sync::Arc;

use ailoy::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    tool::ToolFunc,
};
use napi::{
    JsValue, Status,
    bindgen_prelude::{FromNapiValue, Function, Object, Promise, Unknown},
    sys,
    threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use serde::{Deserialize, de::DeserializeOwned};

use crate::{
    convert::Json,
    model::{Tsfn, async_iterator_symbol, bound},
};

/// Called with the arguments (not error-first). Threadsafe because a JavaScript function runs
/// only on the JavaScript thread, which a turn leaves free as promises iterate it; weak so a
/// registered tool does not keep the process alive.
pub type Callback = ThreadsafeFunction<Json<Value>, ToolReturned, Json<Value>, Status, false, true>;

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
/// model can try differently; from an async generator, it is the last piece.
pub fn tool_func(func: Callback) -> ToolFunc {
    let func = Arc::new(func);
    ToolFunc::new(move |args, id| {
        let func = func.clone();
        Box::pin(async_stream::stream! {
            let returned = match func.call_async_catch(Json(args)).await {
                Ok(returned) => returned,
                Err(e) => {
                    yield output(&id, error(reason(e)));
                    return;
                }
            };
            match returned {
                ToolReturned::Result(result) => match result.settle().await {
                    Ok(value) => yield output(&id, Part::value(value)),
                    Err(e) => yield output(&id, error(e)),
                },
                ToolReturned::Pieces(pieces) => {
                    let mut open = Open(Some(pieces));
                    while let Some(pieces) = &open.0 {
                        let step = match pieces.next.call_async_catch(()).await {
                            Ok(step) => step.settle().await,
                            Err(e) => Err(reason(e)),
                        };
                        match step {
                            Ok(PieceStep { done: true, .. }) => {
                                open.0.take();
                            }
                            Ok(PieceStep { value, .. }) => match value.unwrap_or(Value::Null) {
                                Value::String(text) => yield output(&id, Part::text(text)),
                                value => yield output(&id, Part::value(value)),
                            },
                            Err(e) => {
                                open.0.take();
                                yield output(&id, error(e));
                            }
                        }
                    }
                }
            }
        })
    })
}

fn output(id: &str, part: Part) -> MessageOutput {
    MessageOutput {
        message: Message::new(Role::Tool).with_contents([part]).with_id(id),
        finish_reason: FinishReason::Stop {},
        usage: None,
    }
}

fn error(reason: String) -> Part {
    Part::value(Value::string(format!("error: {reason}")))
}

/// What a tool returned: its result or a promise of it, or an async iterator of its pieces.
pub enum ToolReturned {
    Result(Returned<Value>),
    Pieces(Pieces),
}

impl FromNapiValue for ToolReturned {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let mut kind = sys::ValueType::napi_undefined;
        napi::check_status!(unsafe { sys::napi_typeof(env, napi_val, &mut kind) })?;
        let mut is_promise = false;
        napi::check_status!(unsafe { sys::napi_is_promise(env, napi_val, &mut is_promise) })?;
        if kind == sys::ValueType::napi_object && !is_promise {
            let key = unsafe { async_iterator_symbol(env)? };
            let mut method = std::ptr::null_mut();
            napi::check_status!(unsafe {
                sys::napi_get_property(env, napi_val, key, &mut method)
            })?;
            let mut method_kind = sys::ValueType::napi_undefined;
            napi::check_status!(unsafe { sys::napi_typeof(env, method, &mut method_kind) })?;
            if method_kind == sys::ValueType::napi_function {
                let iterable = unsafe { Object::from_napi_value(env, napi_val)? };
                let method = unsafe { Function::<(), Unknown>::from_napi_value(env, method)? };
                let iterator = method.apply(iterable, ())?;
                let iterator = unsafe { Object::from_napi_value(env, iterator.raw())? };
                return Ok(ToolReturned::Pieces(Pieces::new(iterator)?));
            }
        }
        Ok(ToolReturned::Result(unsafe {
            Returned::from_napi_value(env, napi_val)?
        }))
    }
}

/// An async iterator's result, `{ value, done }`.
#[derive(Deserialize)]
struct PieceStep {
    #[serde(default)]
    done: bool,
    value: Option<Value>,
}

/// The async iterator a tool returned, as its `next` and `return` methods.
pub struct Pieces {
    next: Tsfn<(), Returned<PieceStep>>,
    end: Option<Tsfn<(), Unknown<'static>>>,
}

impl Pieces {
    /// On the JavaScript thread.
    fn new(iterator: Object<'_>) -> napi::Result<Self> {
        let next: Function<(), Returned<PieceStep>> = iterator
            .get("next")?
            .ok_or_else(|| napi::Error::new(Status::InvalidArg, "not an async iterator"))?;
        let end = match iterator.get::<Function<(), Unknown<'static>>>("return")? {
            Some(end) => Some(bound(end, iterator)?),
            None => None,
        };
        Ok(Pieces {
            next: bound(next, iterator)?,
            end,
        })
    }
}

/// An async iterator still open: let go of before its end, its `return()` is called.
struct Open(Option<Pieces>);

impl Drop for Open {
    fn drop(&mut self) {
        if let Some(end) = self.0.take().and_then(|pieces| pieces.end) {
            end.call((), ThreadsafeFunctionCallMode::NonBlocking);
        }
    }
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
