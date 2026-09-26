//! `AgentBuilder`, `Agent`, and the turn an `Agent` runs.
//!
//! # Where the agent lives
//!
//! A Python future has to be `'static`, so it cannot borrow the object it was started from —
//! and [`Agent::run`] borrows the agent for as long as the turn's stream lives. The agent is
//! therefore behind an `Arc<Mutex<..>>`, and a turn holds the lock's *owned* guard inside its
//! stream. Which also makes turns take turns, as `&mut self` does in Rust: a second `run`
//! started while one is being iterated waits for it to finish before it begins.
//!
//! # Where the console lives
//!
//! `AgentBuilder.console` takes a cortex `ConsoleClient` and puts its slot in the agent's state —
//! the slot itself, not what is in it. The `ConsoleClient` object stays usable: its calls and the
//! agent's tools take turns on the one lock, and the agent starts and stops its backend
//! around each batch of tool calls, as the Rust agent does. `ConsoleClient.close()` ends the session
//! for both.
//!
//! # How it ends
//!
//! Dropping an agent may drop the last hold on its console, and a console says `quit` only
//! when it is dropped on a runtime — which a Python finalizer is not on. So [`OnRuntime`]
//! enters the binding's runtime before letting go, for the agent and for a turn's stream
//! alike.

use std::sync::Arc;

use _cortex::console::PyConsoleClient;
use ailoy::{
    agent::{Agent, AgentBuilder, AgentSpec, AgentState, ContextManager},
    datatype::Value,
    lang_model::ResponseFormat,
    memory::Memory,
    message::Message,
    tool::{ToolDesc, WebSearchEngineKind},
};
use futures::{StreamExt as _, stream::BoxStream};
use pyo3::{
    exceptions::{PyStopAsyncIteration, PyValueError},
    prelude::*,
};
use pyo3_async_runtimes::tokio::{future_into_py, get_runtime};
use tokio::sync::Mutex;

use crate::{
    convert::{Query, from_py, to_py},
    error::{self, AiloyError},
};

/// A value let go of on the binding's runtime — see the module docs.
struct OnRuntime<T>(Option<T>);

impl<T> Drop for OnRuntime<T> {
    fn drop(&mut self) {
        let _entered = get_runtime().enter();
        self.0.take();
    }
}

/// An [`AgentBuilder`], filled in place and emptied by `build()`.
///
/// In place rather than by value, as cortex's `ConsoleClientBuilder` is: the Rust builder is
/// consumed by each call and is not `Clone`, so there is exactly one of it to hand along.
/// Each method returns the same object so calls chain as they do in Rust.
#[pyclass(name = "AgentBuilder", module = "ailoy")]
pub struct PyAgentBuilder(std::sync::Mutex<Option<AgentBuilder>>);

impl PyAgentBuilder {
    fn update<'py>(
        slf: PyRef<'py, Self>,
        f: impl FnOnce(AgentBuilder) -> PyResult<AgentBuilder>,
    ) -> PyResult<PyRef<'py, Self>> {
        {
            let mut held = slf.0.lock().unwrap();
            let builder = held.take().ok_or_else(built)?;
            *held = Some(f(builder)?);
        }
        Ok(slf)
    }
}

fn built() -> PyErr {
    PyValueError::new_err("this AgentBuilder has already been built")
}

#[pymethods]
impl PyAgentBuilder {
    /// A builder for `model`, such as `"anthropic/claude-sonnet-5"`: a name some registered
    /// language-model provider serves.
    #[new]
    fn new(model: String) -> Self {
        PyAgentBuilder(std::sync::Mutex::new(Some(AgentBuilder::new(model))))
    }

    fn agent_provider(slf: PyRef<'_, Self>, name: String) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.agent_provider(name)))
    }

    fn instruction(slf: PyRef<'_, Self>, instruction: String) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.instruction(instruction)))
    }

    fn tool<'py>(slf: PyRef<'py, Self>, desc: Bound<'py, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let desc: ToolDesc = from_py(&desc, "a tool description")?;
        Self::update(slf, |b| Ok(b.tool(desc)))
    }

    fn tools<'py>(slf: PyRef<'py, Self>, descs: Bound<'py, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let descs: Vec<ToolDesc> = from_py(&descs, "a list of tool descriptions")?;
        Self::update(slf, |b| Ok(b.tools(descs)))
    }

    fn system_tools(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.system_tools()))
    }

    fn shell_tool(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.shell_tool()))
    }

    /// `engines` by name — `"Google"`, `"DuckDuckGo"`, … — or every engine when empty.
    #[pyo3(signature = (engines = None))]
    fn web_search_tool<'py>(
        slf: PyRef<'py, Self>,
        engines: Option<Bound<'py, PyAny>>,
    ) -> PyResult<PyRef<'py, Self>> {
        let engines: Vec<WebSearchEngineKind> = match engines {
            Some(engines) => from_py(&engines, "a list of web search engines")?,
            None => Vec::new(),
        };
        Self::update(slf, |b| Ok(b.web_search_tool(engines)))
    }

    fn web_fetch_tool(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.web_fetch_tool()))
    }

    /// A sub-agent, as the dict an `AgentSpec` serializes to. It has to carry a `card`.
    fn subagent<'py>(slf: PyRef<'py, Self>, spec: Bound<'py, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let spec: AgentSpec = from_py(&spec, "an agent spec")?;
        Self::update(slf, |b| Ok(b.subagent(spec)))
    }

    fn history<'py>(
        slf: PyRef<'py, Self>,
        history: Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let history: Vec<Message> = from_py(&history, "a list of messages")?;
        Self::update(slf, |b| Ok(b.history(history)))
    }

    /// Run the agent's console tools in `console`'s session, which the agent then shares.
    fn console<'py>(
        slf: PyRef<'py, Self>,
        console: PyRef<'py, PyConsoleClient>,
    ) -> PyResult<PyRef<'py, Self>> {
        let slot = console.slot();
        Self::update(slf, |b| Ok(b.shared_console(slot)))
    }

    /// Remember into the memory store at `memfile`, a path in the console.
    fn memory(slf: PyRef<'_, Self>, memfile: String) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.memory(Memory::new(memfile))))
    }

    fn skill(slf: PyRef<'_, Self>, dir: String) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.skill(dir)))
    }

    #[pyo3(signature = (max_input_tokens = None, preserve_recent_turns = None))]
    fn context_manager(
        slf: PyRef<'_, Self>,
        max_input_tokens: Option<u64>,
        preserve_recent_turns: Option<usize>,
    ) -> PyResult<PyRef<'_, Self>> {
        let default = ContextManager::default();
        let cm = ContextManager {
            max_input_tokens: max_input_tokens.unwrap_or(default.max_input_tokens),
            preserve_recent_turns: preserve_recent_turns.unwrap_or(default.preserve_recent_turns),
        };
        Self::update(slf, |b| Ok(b.context_manager(cm)))
    }

    fn max_tokens(slf: PyRef<'_, Self>, max_tokens: u64) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.max_tokens(max_tokens)))
    }

    fn temperature(slf: PyRef<'_, Self>, temperature: f64) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.temperature(temperature)))
    }

    fn top_p(slf: PyRef<'_, Self>, top_p: f64) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.top_p(top_p)))
    }

    fn top_k(slf: PyRef<'_, Self>, top_k: u64) -> PyResult<PyRef<'_, Self>> {
        Self::update(slf, |b| Ok(b.top_k(top_k)))
    }

    /// Constrain replies to the JSON schema `schema`, which is checked here.
    fn response_format<'py>(
        slf: PyRef<'py, Self>,
        schema: Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let schema: Value = from_py(&schema, "a JSON schema")?;
        let fmt = ResponseFormat::json_schema(schema)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        Self::update(slf, |b| Ok(b.response_format(fmt)))
    }

    /// Make the agent — an awaitable, as `AgentBuilder::build` is a future: it reads the
    /// skills through the console.
    fn build<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let builder = self.0.lock().unwrap().take().ok_or_else(built)?;
        future_into_py(py, async move {
            let agent = builder.build().await.map_err(error::anyhow)?;
            Ok(PyAgent::new(agent))
        })
    }
}

type Held = Arc<Mutex<OnRuntime<Agent>>>;

#[pyclass(name = "Agent", module = "ailoy", frozen)]
pub struct PyAgent {
    agent: Held,
}

impl PyAgent {
    fn new(agent: Agent) -> Self {
        PyAgent {
            agent: Arc::new(Mutex::new(OnRuntime(Some(agent)))),
        }
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
fn item<T: serde::Serialize>(item: anyhow::Result<T>) -> PyResult<Py<PyAny>> {
    let item = item.map_err(error::anyhow)?;
    Python::attach(|py| Ok(to_py(py, &item)?.unbind()))
}

/// One turn's stream, from the agent's owned guard: `'static`, so a Python future can hold it.
fn turn(agent: Held, query: Message, mode: Mode) -> BoxStream<'static, PyResult<Py<PyAny>>> {
    async_stream::stream! {
        let mut guard = agent.lock_owned().await;
        let Some(agent) = guard.0.as_mut() else {
            yield Err(closed());
            return;
        };
        match mode {
            Mode::Messages => {
                let mut stream = agent.run(query);
                while let Some(output) = stream.next().await {
                    yield item(output);
                }
            }
            Mode::Deltas => {
                let mut stream = agent.run_stream(query);
                while let Some(delta) = stream.next().await {
                    yield item(delta);
                }
            }
        }
    }
    .boxed()
}

#[pymethods]
impl PyAgent {
    /// An agent from a spec — the dict an `AgentSpec` serializes to — for a caller who holds
    /// one rather than building it up. Awaitable.
    #[staticmethod]
    #[pyo3(signature = (spec, *, agent_provider = "default".to_string(), history = None, console = None, memory = None))]
    fn from_spec<'py>(
        py: Python<'py>,
        spec: Bound<'py, PyAny>,
        agent_provider: String,
        history: Option<Bound<'py, PyAny>>,
        console: Option<PyRef<'py, PyConsoleClient>>,
        memory: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec: AgentSpec = from_py(&spec, "an agent spec")?;
        let mut state = AgentState::new();
        if let Some(history) = history {
            state = state.with_history(from_py::<Vec<Message>>(&history, "a list of messages")?);
        }
        if let Some(console) = console {
            state = state.with_console_slot(console.slot());
        }
        if let Some(memfile) = memory {
            state = state.with_memory(Memory::new(memfile));
        }
        future_into_py(py, async move {
            let agent = Agent::try_with_provider_and_state(spec, &agent_provider, state)
                .await
                .map_err(error::anyhow)?;
            Ok(PyAgent::new(agent))
        })
    }

    /// The conversation so far, as a list of messages.
    ///
    /// Read without waiting, so it raises while a turn holds the agent rather than making
    /// this property a coroutine.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let guard = self
            .agent
            .try_lock()
            .map_err(|_| AiloyError::new_err("this agent is in the middle of a turn"))?;
        let agent = guard.0.as_ref().ok_or_else(closed)?;
        to_py(py, &agent.get_history())
    }

    /// Run one turn on `query` — a message dict, or a string as the user's text — and
    /// iterate what it produces: each complete message, the model's and each tool's.
    fn run(&self, query: Query) -> PyAgentRun {
        PyAgentRun::new(turn(self.agent.clone(), query.0, Mode::Messages))
    }

    /// Run one turn as `run` does, but iterate the model's output as it is generated: deltas
    /// for each message the model writes, and each tool's result whole.
    fn run_stream(&self, query: Query) -> PyAgentRun {
        PyAgentRun::new(turn(self.agent.clone(), query.0, Mode::Deltas))
    }

    /// Let go of the agent now, and of its console if nothing else holds it. Waits for a turn
    /// being iterated to finish. Closing twice is the same as closing once.
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let agent = self.agent.clone();
        future_into_py(py, async move {
            // Dropped here, on the runtime.
            agent.lock().await.0.take();
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
    fn new(stream: BoxStream<'static, PyResult<Py<PyAny>>>) -> Self {
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
    m.add_class::<PyAgentBuilder>()?;
    m.add_class::<PyAgent>()?;
    m.add_class::<PyAgentRun>()?;
    Ok(())
}
