//! `Agent`, and the turn an `Agent` runs: [`ailoy::Agent`], reached over HTTP from a server in
//! this process.
//!
//! A Python future must be `'static`, but a turn's stream borrows the agent. So the agent sits
//! behind an `Arc`, which each turn holds a clone of. The server runs one turn of an agent at a
//! time and refuses another while one runs.
//!
//! `Agent.from_spec`'s `sandbox` gives the agent a `virtx.Sandbox` (see [`crate::console`]),
//! which stays the caller's: the agent runs in it over the same client, and never removes it.

use std::sync::Arc;

use ailoy::{Agent, agent::AgentSpec, message::Message};
use futures::{StreamExt as _, stream::BoxStream};
use pyo3::{exceptions::PyStopAsyncIteration, prelude::*};
use pyo3_async_runtimes::tokio::{future_into_py, get_runtime};
use tokio::sync::Mutex;

use crate::{
    console,
    convert::{Query, from_py, to_py},
    error::{self, AiloyError},
};

/// A value let go of on the binding's runtime: what an agent or a turn's stream holds may let
/// go of itself on a task, and a Python finalizer is not on a runtime.
pub struct OnRuntime<T>(pub Option<T>);

impl<T> Drop for OnRuntime<T> {
    fn drop(&mut self) {
        let _entered = get_runtime().enter();
        self.0.take();
    }
}

type Held = Arc<OnRuntime<Agent>>;

#[pyclass(name = "Agent", module = "ailoy", frozen)]
pub struct PyAgent {
    /// None once closed.
    agent: std::sync::Mutex<Option<Held>>,
}

impl PyAgent {
    fn new(agent: Agent) -> Self {
        PyAgent {
            agent: std::sync::Mutex::new(Some(Arc::new(OnRuntime(Some(agent))))),
        }
    }

    fn held(&self) -> PyResult<Held> {
        self.agent.lock().unwrap().clone().ok_or_else(closed)
    }

    /// The turn's stream; one that raises at once, for an agent closed.
    fn start(&self, query: Message, mode: Mode) -> PyAgentRun {
        let stream = match self.held() {
            Ok(agent) => turn(agent, query, mode),
            Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
        };
        PyAgentRun::new(stream)
    }
}

fn closed() -> PyErr {
    AiloyError::new_err("this agent has been closed")
}

/// Which of the agent's two streams a turn is iterated from.
#[derive(Clone, Copy)]
enum Mode {
    Messages,
    Deltas,
}

/// A turn's item as the dict it serializes to, or the error it ended on.
pub fn item<T: serde::Serialize>(item: anyhow::Result<T>) -> PyResult<Py<PyAny>> {
    let item = item.map_err(error::anyhow)?;
    Python::attach(|py| Ok(to_py(py, &item)?.unbind()))
}

/// One turn's stream, holding the agent: `'static`, so a Python future can hold it.
fn turn(agent: Held, query: Message, mode: Mode) -> BoxStream<'static, PyResult<Py<PyAny>>> {
    async_stream::stream! {
        let agent = agent.0.as_ref().expect("held until dropped");
        match mode {
            Mode::Messages => {
                let mut stream = std::pin::pin!(agent.run(&query));
                while let Some(output) = stream.next().await {
                    yield item(output);
                }
            }
            Mode::Deltas => {
                let mut stream = std::pin::pin!(agent.run_stream(&query));
                while let Some(delta) = stream.next().await {
                    yield item(delta.map_err(anyhow::Error::from));
                }
            }
        }
    }
    .boxed()
}

#[pymethods]
impl PyAgent {
    /// An agent from a spec, the dict an `AgentSpec` serializes to, with the history it
    /// starts with and the `virtx.Sandbox` it works in. Awaitable.
    #[staticmethod]
    #[pyo3(signature = (spec, *, history = None, sandbox = None))]
    fn from_spec<'py>(
        py: Python<'py>,
        spec: Bound<'py, PyAny>,
        history: Option<Bound<'py, PyAny>>,
        sandbox: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec: AgentSpec = from_py(&spec, "an agent spec")?;
        let history = match history {
            Some(history) => from_py::<Vec<Message>>(&history, "a list of messages")?,
            None => Vec::new(),
        };
        let sandbox = sandbox
            .map(|sandbox| console::sandbox(&sandbox))
            .transpose()?;
        future_into_py(py, async move {
            let agent = Agent::from_spec(spec, history, sandbox.as_ref())
                .await
                .map_err(|e| error::anyhow(e.into()))?;
            Ok(PyAgent::new(agent))
        })
    }

    /// The conversation so far, as a list of messages, as the server keeps it. Awaitable.
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let agent = self.held()?;
        future_into_py(py, async move {
            let agent = agent.0.as_ref().expect("held until dropped");
            let history = agent.history().await.map_err(|e| error::anyhow(e.into()))?;
            Python::attach(|py| Ok(to_py(py, &history)?.unbind()))
        })
    }

    /// Run one turn on `query` — a message dict, or a string as the user's text — and
    /// iterate what it produces: each complete message, the model's and each tool's.
    fn run(&self, query: Query) -> PyAgentRun {
        self.start(query.0, Mode::Messages)
    }

    /// Run one turn as `run` does, but iterate the model's output as it is generated: deltas
    /// for each message the model writes, and each tool's result whole.
    fn run_stream(&self, query: Query) -> PyAgentRun {
        self.start(query.0, Mode::Deltas)
    }

    /// Let go of the agent and its session on the server, and of its hold of its sandbox's
    /// client. A turn being iterated still ends. Closing twice is the same as closing once.
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let agent = self.agent.lock().unwrap().take();
        future_into_py(py, async move {
            if let Some(agent) = agent {
                let closing = agent.0.as_ref().expect("held until dropped").close().await;
                closing.map_err(|e| error::anyhow(e.into()))?;
            }
            Ok(())
        })
    }

    fn __aenter__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let slf = slf.unbind();
        future_into_py(py, async move { Ok(slf) })
    }

    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _exc_type: Bound<'py, PyAny>,
        _exc: Bound<'py, PyAny>,
        _tb: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.close(py)
    }
}

type Stream = Arc<Mutex<OnRuntime<BoxStream<'static, PyResult<Py<PyAny>>>>>>;

/// A turn, iterated with `async for`.
///
/// Nothing runs until the first `__anext__`, and the turn only goes as far as it has been
/// iterated: breaking out of the loop, or `aclose()`, ends it where it stands.
#[pyclass(name = "AgentRun", module = "ailoy", frozen)]
pub struct PyAgentRun {
    stream: Stream,
}

impl PyAgentRun {
    pub fn new(stream: BoxStream<'static, PyResult<Py<PyAny>>>) -> Self {
        PyAgentRun {
            stream: Arc::new(Mutex::new(OnRuntime(Some(stream)))),
        }
    }
}

#[pymethods]
impl PyAgentRun {
    fn __aiter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        future_into_py(py, async move {
            let mut guard = stream.lock().await;
            let Some(stream) = guard.0.as_mut() else {
                return Err(PyStopAsyncIteration::new_err(()));
            };
            match stream.next().await {
                Some(item) => item,
                None => {
                    // Let go of the agent now rather than when this object is collected.
                    guard.0.take();
                    Err(PyStopAsyncIteration::new_err(()))
                }
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        future_into_py(py, async move {
            stream.lock().await.0.take();
            Ok(())
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgent>()?;
    m.add_class::<PyAgentRun>()?;
    Ok(())
}
