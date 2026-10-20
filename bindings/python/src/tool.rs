//! A Python callable as a [`ToolFunc`].
//!
//! The model's arguments, a dict whose keys the tool's parameters schema names, are passed as
//! keyword arguments; non-object arguments as the one positional argument. The callable returns
//! the result or an awaitable of it, and the result must be what a message can hold: `None`, a
//! bool, a number, a string, or a list or dict of those.
//!
//! An async generator streams its result instead: each value it yields is a piece of it, a
//! string as text and anything else as a value, and its end finishes the result. A turn let go
//! of mid-call closes the generator, so its `finally` runs.
//!
//! A Python tool is not handed the console; one that runs commands holds its own
//! `ConsoleClient`.

use std::{ffi::CStr, sync::Arc};

use ailoy::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    tool::ToolFunc,
};
use futures::{StreamExt as _, stream::BoxStream};
use pyo3::{
    prelude::*,
    sync::PyOnceLock,
    types::{PyCFunction, PyDict, PyTuple},
};
use pyo3_async_runtimes::{TaskLocals, into_future_with_locals, tokio::get_current_locals};
use tokio::sync::oneshot;

use crate::convert::{from_py, to_py};

/// A raise becomes the result `"error: ValueError: ..."` instead of ending the turn, so the
/// model can try differently; from an async generator, it is the last piece.
pub fn tool_func(func: Py<PyAny>) -> ToolFunc {
    let func = Arc::new(func);
    ToolFunc::new(move |args, id| {
        let func = func.clone();
        Box::pin(async_stream::stream! {
            match call(func, args).await {
                Ok(Returned::Value(value)) => yield output(&id, Part::value(value)),
                Ok(Returned::Pieces(mut pieces)) => {
                    while let Some(piece) = pieces.next().await {
                        match piece {
                            Ok(Value::String(text)) => yield output(&id, Part::text(text)),
                            Ok(value) => yield output(&id, Part::value(value)),
                            Err(e) => {
                                yield output(&id, error(e));
                                break;
                            }
                        }
                    }
                }
                Err(e) => yield output(&id, error(e)),
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

fn error(e: PyErr) -> Part {
    Part::value(Value::string(format!("error: {e}")))
}

/// What a tool returned: its result, or the pieces of it an async generator yields.
enum Returned {
    Value(Value),
    Pieces(BoxStream<'static, PyResult<Value>>),
}

async fn call(func: Arc<Py<PyAny>>, args: Value) -> PyResult<Returned> {
    let locals: Option<TaskLocals> = Python::attach(|py| get_current_locals(py).ok());
    let returned = call_py(move |py| {
        let args = to_py(py, &args)?;
        match args.cast::<PyDict>() {
            Ok(kwargs) => func.call(py, (), Some(kwargs)),
            Err(_) => func.call1(py, (args,)),
        }
    })
    .await?;
    Python::attach(|py| {
        let returned = returned.bind(py);
        if returned.hasattr("__anext__")? && returned.hasattr("aclose")? {
            let steps = steps_class(py)?
                .call1((returned, event_loop(locals.as_ref())?.event_loop(py)))?
                .unbind();
            return Ok(Returned::Pieces(pieces(steps).boxed()));
        }
        from_py(returned, "a tool result a message can hold").map(Returned::Value)
    })
}

/// Steps an async generator on its event loop for a caller on another thread. `Steps.next`
/// runs one `__anext__` as a task, and `Steps.close` cancels a step under way and closes the
/// generator, so its `finally` runs.
const STEPS: &CStr = cr#"
import asyncio

_closing = set()


class Steps:
    def __init__(self, agen, loop):
        self.agen = agen
        self.loop = loop
        self.task = None

    def next(self, done):
        self.loop.call_soon_threadsafe(self._next, done)

    def _next(self, done):
        self.task = self.loop.create_task(self._anext())
        self.task.add_done_callback(done)

    async def _anext(self):
        try:
            return (await self.agen.__anext__(),)
        except StopAsyncIteration:
            return ()

    def close(self):
        self.loop.call_soon_threadsafe(self._close)

    def _close(self):
        task = self.loop.create_task(self._aclose())
        _closing.add(task)
        task.add_done_callback(_closing.discard)

    async def _aclose(self):
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except BaseException:
                pass
        await self.agen.aclose()
"#;

fn steps_class(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    CLASS
        .get_or_try_init(py, || {
            let module = PyModule::from_code(py, STEPS, c"ailoy_steps.py", c"ailoy_steps")?;
            Ok::<_, PyErr>(module.getattr("Steps")?.unbind())
        })
        .map(|class| class.bind(py))
}

/// What `steps` yields, one step at a time.
fn pieces(steps: Py<PyAny>) -> impl futures::Stream<Item = PyResult<Value>> + Send + 'static {
    async_stream::stream! {
        let mut open = Open(Some(steps));
        while let Some(steps) = &open.0 {
            match step(steps).await {
                Ok(Some(value)) => yield Ok(value),
                Ok(None) => {
                    open.0.take();
                }
                Err(e) => {
                    open.0.take();
                    yield Err(e);
                }
            }
        }
    }
}

/// The next value of `steps`, or `None` at the generator's end.
async fn step(steps: &Py<PyAny>) -> PyResult<Option<Value>> {
    let (tx, rx) = oneshot::channel();
    let tx = std::sync::Mutex::new(Some(tx));
    Python::attach(|py| {
        let done = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
            let result = args.get_item(0)?.call_method0("result").map(Bound::unbind);
            if let Some(tx) = tx.lock().unwrap().take() {
                let _ = tx.send(result);
            }
            Ok(())
        })?;
        steps.call_method1(py, "next", (done,))
    })?;
    let item = rx.await.map_err(|_| {
        crate::error::AiloyError::new_err("the tool's event loop closed mid-step")
    })??;
    Python::attach(|py| {
        let item = item.bind(py).cast::<PyTuple>()?;
        match item.len() {
            0 => Ok(None),
            _ => from_py(&item.get_item(0)?, "a piece a tool result can hold").map(Some),
        }
    })
}

/// An async generator still open: let go of before its end, it is closed.
struct Open(Option<Py<PyAny>>);

impl Drop for Open {
    fn drop(&mut self) {
        if let Some(steps) = self.0.take() {
            // A closed event loop has nothing left to run the generator on.
            Python::attach(|py| {
                let _ = steps.call_method0(py, "close");
            });
        }
    }
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
