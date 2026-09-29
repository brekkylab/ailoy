//! A Python callable as a [`ToolFunc`].
//!
//! The model's arguments, a dict whose keys the tool's parameters schema names, are passed as
//! keyword arguments; non-object arguments are passed as the one positional argument. The
//! return value becomes the tool's result, so it must be what a message can hold: `None`, a
//! bool, a number, a string, or a list or dict of those.
//!
//! # Sync and async
//!
//! A plain return value is the result. An awaitable is awaited on the event loop the turn is
//! iterated from, which `pyo3-async-runtimes` puts in the task locals of the `__anext__` the
//! call happens inside.
//!
//! The call runs on tokio's blocking pool, not a worker, so a slow synchronous tool does not
//! starve the rest of the turn or tools running beside it.
//!
//! # Failure
//!
//! A raise becomes the result `"error: ValueError: ..."` instead of ending the turn, so the
//! model can try differently.
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
                depth: None,
                source_agent: None,
            }
        })
        .boxed()
    })
}

async fn call(func: Arc<Py<PyAny>>, args: Value) -> PyResult<Value> {
    // Taken before the blocking pool, whose threads are outside the task. A turn driven from
    // outside an event loop has none, which only matters for a tool returning an awaitable.
    let locals: Option<TaskLocals> = Python::attach(|py| get_current_locals(py).ok());

    let returned = tokio::task::spawn_blocking(move || {
        Python::attach(|py| -> PyResult<Py<PyAny>> {
            let args = to_py(py, &args)?;
            let returned = match args.cast::<PyDict>() {
                Ok(kwargs) => func.call(py, (), Some(kwargs))?,
                Err(_) => func.call1(py, (args,))?,
            };
            Ok(returned)
        })
    })
    .await
    .map_err(|e| crate::error::AiloyError::new_err(format!("the tool did not finish: {e}")))??;

    let awaited = Python::attach(|py| -> PyResult<_> {
        let returned = returned.bind(py);
        if !returned.hasattr("__await__")? {
            return Ok(None);
        }
        let locals = locals.as_ref().ok_or_else(|| {
            crate::error::AiloyError::new_err(
                "an async tool needs its turn iterated from a running event loop",
            )
        })?;
        Ok(Some(into_future_with_locals(locals, returned.clone())?))
    })?;
    let returned = match awaited {
        Some(fut) => fut.await?,
        None => returned,
    };

    Python::attach(|py| from_py(returned.bind(py), "a tool result a message can hold"))
}
