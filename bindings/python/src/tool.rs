//! A Python callable as a [`ToolFunc`].
//!
//! The model's arguments, a dict whose keys the tool's parameters schema names, are passed as
//! keyword arguments; non-object arguments as the one positional argument. The callable returns
//! the result or an awaitable of it, and the result must be what a message can hold: `None`, a
//! bool, a number, a string, or a list or dict of those.
//!
//! A Python tool is not handed the console; one that runs commands holds its own
//! `ConsoleClient`.

use std::sync::Arc;

use ailoy::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    tool::ToolFunc,
};
use futures::StreamExt as _;
use pyo3::{prelude::*, types::PyDict};
use pyo3_async_runtimes::{TaskLocals, into_future_with_locals, tokio::get_current_locals};

use crate::convert::{from_py, to_py};

/// A raise becomes the result `"error: ValueError: ..."` instead of ending the turn, so the
/// model can try differently.
pub fn tool_func(func: Py<PyAny>) -> ToolFunc {
    let func = Arc::new(func);
    ToolFunc::new(move |args, id| {
        let func = func.clone();
        futures::stream::once(async move {
            let value = call(func, args)
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

async fn call(func: Arc<Py<PyAny>>, args: Value) -> PyResult<Value> {
    let returned = call_py(move |py| {
        let args = to_py(py, &args)?;
        match args.cast::<PyDict>() {
            Ok(kwargs) => func.call(py, (), Some(kwargs)),
            Err(_) => func.call1(py, (args,)),
        }
    })
    .await?;
    Python::attach(|py| from_py(returned.bind(py), "a tool result a message can hold"))
}

/// Call into Python with `f` and await what it returns if that is awaitable.
///
/// On the blocking pool, so a slow synchronous callable does not starve the turn or the
/// futures beside it. An awaitable is awaited on the turn's event loop, which
/// `pyo3-async-runtimes` puts in the task locals of the enclosing `__anext__`; they are taken
/// before the blocking pool, whose threads are outside the task. A turn driven from outside an
/// event loop has none, which only matters for a callable returning an awaitable.
pub async fn call_py(
    f: impl FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
) -> PyResult<Py<PyAny>> {
    let locals: Option<TaskLocals> = Python::attach(|py| get_current_locals(py).ok());

    let returned = tokio::task::spawn_blocking(move || Python::attach(f))
        .await
        .map_err(|e| {
            crate::error::AiloyError::new_err(format!("the call did not finish: {e}"))
        })??;

    let awaited = Python::attach(|py| -> PyResult<_> {
        let returned = returned.bind(py);
        if !returned.hasattr("__await__")? {
            return Ok(None);
        }
        Ok(Some(into_future_with_locals(
            &event_loop(locals.as_ref())?,
            returned.clone(),
        )?))
    })?;
    match awaited {
        Some(fut) => fut.await,
        None => Ok(returned),
    }
}

/// The turn's event loop, or the error saying an async callable needs one.
pub fn event_loop(locals: Option<&TaskLocals>) -> PyResult<TaskLocals> {
    locals.cloned().ok_or_else(|| {
        crate::error::AiloyError::new_err(
            "an async callable needs its turn iterated from a running event loop",
        )
    })
}
