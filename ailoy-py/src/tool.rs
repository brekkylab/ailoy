//! A Python callable as a [`ToolFunc`].
//!
//! The model's arguments, a dict whose keys the tool's parameters schema names, are passed as
//! keyword arguments; non-object arguments as the one positional argument. The callable returns
//! the result or an awaitable of it, and the result must be what a message can hold: `None`, a
//! bool, a number, a string, or a list or dict of those.
//!
//! A Python tool is not handed the sandbox; one that runs commands holds its own
//! `virtx.Sandbox`.

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
/// model can try differently. An awaitable it returns is awaited on `locals`' event loop, the
/// one running where it was registered.
pub fn tool_func(func: Py<PyAny>, locals: Option<TaskLocals>) -> ToolFunc {
    let func = Arc::new(func);
    ToolFunc::new(move |args, id| {
        let (func, locals) = (func.clone(), locals.clone());
        futures::stream::once(async move {
            let value = call(func, args, locals)
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

async fn call(func: Arc<Py<PyAny>>, args: Value, locals: Option<TaskLocals>) -> PyResult<Value> {
    let returned = call_py(
        move |py| {
            let args = to_py(py, &args)?;
            match args.cast::<PyDict>() {
                Ok(kwargs) => func.call(py, (), Some(kwargs)),
                Err(_) => func.call1(py, (args,)),
            }
        },
        locals,
    )
    .await?;
    Python::attach(|py| from_py(returned.bind(py), "a tool result a message can hold"))
}

/// Call into Python with `f` and await what it returns if that is awaitable.
///
/// On the blocking pool, so a slow synchronous callable does not starve the turn or the
/// futures beside it. An awaitable is awaited on the event loop of the task locals here, which
/// `pyo3-async-runtimes` puts in those of an enclosing `__anext__`, else on `registered`'s, the
/// loop running where the callable was registered: a turn runs on a task of the server's own.
/// The locals are taken before the blocking pool, whose threads are outside the task.
pub async fn call_py(
    f: impl FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
    registered: Option<TaskLocals>,
) -> PyResult<Py<PyAny>> {
    let here: Option<TaskLocals> = Python::attach(|py| get_current_locals(py).ok());
    let locals = here.or(registered);

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
            "an async callable needs an event loop running where it was registered",
        )
    })
}
