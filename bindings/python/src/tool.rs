//! A Python callable as a [`ToolFunc`].
//!
//! The model's arguments arrive as a dict and are passed as keyword arguments — the
//! parameters schema a tool is described by names them, so they are the callable's
//! parameters. Arguments that are not an object are passed as the one positional argument.
//! What the callable returns becomes the tool's result value, so it has to be what a
//! message can hold: `None`, a bool, a number, a string, or a list or dict of those.
//!
//! # Sync and async
//!
//! Either is taken, and which is told by what calling it returns. A plain result is the
//! answer. An awaitable is awaited on the event loop the turn is being iterated from — the
//! one `pyo3-async-runtimes` puts in the task's locals for every future it runs, and so for
//! the `__anext__` a tool call happens inside.
//!
//! The call itself is made on tokio's blocking pool, not on a worker: a synchronous tool may
//! take as long as it likes, and a worker it held would be one the rest of the turn — and
//! any other tool running beside it — could not use.
//!
//! # Failure
//!
//! A tool that raises answers with the exception as its result, `"error: ValueError: ..."`,
//! rather than ending the turn: the model asked for something that did not work, and it is
//! the one that can try differently.
//!
//! A Python tool is pure — it is not handed the console. One that needs to run a command
//! does it through a `ConsoleClient` it holds itself.

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
    // Taken here, before the blocking pool: the locals are the task's, and a blocking thread
    // is not in the task. A turn driven from outside an event loop has none, which only a
    // tool that returns an awaitable finds out.
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
