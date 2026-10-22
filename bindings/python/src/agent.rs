//! `AgentBuilder`, `Agent`, and the turn an `Agent` runs.
//!
//! A Python future must be `'static`, but [`Agent::run`] borrows the agent for as long as the
//! turn's stream lives. So the agent sits behind an `Arc<Mutex<..>>` and a turn holds the
//! lock's owned guard in its stream, which also makes a second `run` wait for the first to
//! finish, as `&mut self` would in Rust.
//!
//! `AgentBuilder.console` attaches the agent to a `virtx.ConsoleClient`'s session (see
//! [`crate::console`]). The `ConsoleClient` stays usable: its calls and the agent's take turns
//! on its lock, one protocol call at a time, the agent starts and stops its backend around each
//! batch of tool calls, and `ConsoleClient.close()` ends the session for both.

use std::{path::PathBuf, sync::Arc};

use ailoy::{
    agent::{Agent, AgentBuilder, AgentSpec, AgentState, ContextManager},
    browser::{Browser, UserBrowser},
    datatype::Value,
    memory::Memory,
    message::Message,
    model::ThinkingEffort,
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
    console,
    convert::{Query, from_py, to_py},
    error::{self, AiloyError},
};

/// A value let go of on the binding's runtime. Dropping an agent or a turn's stream may drop
/// the last hold on its console, whose client is let go of on a task when dropped on a runtime
/// (an attached console says no `quit`: the session is the `virtx.ConsoleClient`'s), and a
/// Python finalizer is not on one.
pub struct OnRuntime<T>(pub Option<T>);

impl<T> Drop for OnRuntime<T> {
    fn drop(&mut self) {
        let _entered = get_runtime().enter();
        self.0.take();
    }
}

/// An [`AgentBuilder`], filled in place since the Rust one is consumed per call and not
/// `Clone`, and emptied by `build()`. Methods return `self` to chain.
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

/// A user browser as Python names it: a name, or a path to any other's user data directory. A
/// string is a name unless it looks like a path, so a misspelt name is refused, not looked for.
fn user_browser_of(value: &Bound<'_, PyAny>) -> PyResult<UserBrowser> {
    if let Ok(name) = value.extract::<String>() {
        if let Some(named) = UserBrowser::named(&name) {
            return Ok(named);
        }
        if !name.contains(['/', '\\']) {
            let names: Vec<&str> = UserBrowser::NAMED.iter().map(UserBrowser::name).collect();
            return Err(PyValueError::new_err(format!(
                "user_browser is one of {} or a user data directory, not {name:?}",
                names.join(", ")
            )));
        }
        return Ok(UserBrowser::Custom(name.into()));
    }
    let dir: PathBuf = value.extract().map_err(|_| {
        PyValueError::new_err("user_browser is a browser's name or its user data directory")
    })?;
    Ok(UserBrowser::Custom(dir))
}

#[pymethods]
impl PyAgentBuilder {
    /// A builder for `model`, such as `"anthropic/claude-sonnet-5-5"`: `<org>/<name>`, built
    /// through the provider set for `<org>`.
    #[new]
    fn new(model: String) -> Self {
        PyAgentBuilder(std::sync::Mutex::new(Some(AgentBuilder::new(model))))
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

    fn history<'py>(
        slf: PyRef<'py, Self>,
        history: Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let history: Vec<Message> = from_py(&history, "a list of messages")?;
        Self::update(slf, |b| Ok(b.history(history)))
    }

    /// Run the agent's console tools in the session of `console`, a `virtx.ConsoleClient`,
    /// which the agent then shares.
    fn console<'py>(
        slf: PyRef<'py, Self>,
        console: Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let slot = console::attach(&console)?;
        Self::update(slf, |b| Ok(b.shared_console(slot)))
    }

    /// A browser the agent drives with its `browser_*` tools over CDP from this process rather
    /// than the console. By default the person's own Chromium-based browser with remote
    /// debugging on (Chrome 144 or later by default), once the person allows it; with
    /// `isolated`, one of the agent's own, started on first use and closed with the agent: in a
    /// window unless `headless`, `executable` if given or else a Chromium-based browser installed
    /// here, on the user data directory `profile` if given or else a fresh one.
    /// `user_browser` names which of the person's:
    /// `"chrome"`, `"chrome-beta"`, `"chrome-dev"`, `"chrome-canary"`, `"chromium"`, `"edge"`,
    /// `"brave"`, or any other by its user data directory (a path); with none, whichever turned
    /// remote debugging on last. `careful` tells the agent to use sites as a person
    /// would, so sites checking for bots are less likely to turn it away. `start_url` is the page
    /// the agent's tab opens on.
    #[pyo3(signature = (*, user_browser = None, careful = true, start_url = None, isolated = false, headless = false, executable = None, profile = None))]
    #[allow(clippy::too_many_arguments)]
    fn browser<'py>(
        slf: PyRef<'py, Self>,
        user_browser: Option<Bound<'py, PyAny>>,
        careful: bool,
        start_url: Option<String>,
        isolated: bool,
        headless: bool,
        executable: Option<PathBuf>,
        profile: Option<PathBuf>,
    ) -> PyResult<PyRef<'py, Self>> {
        let mut browser = if isolated {
            Browser::isolated()
        } else {
            Browser::user()
        };
        if let Some(host) = user_browser {
            browser = browser.user_browser(user_browser_of(&host)?);
        }
        if headless {
            browser = browser.headless(true);
        }
        if let Some(path) = executable {
            browser = browser.executable(path);
        }
        if let Some(dir) = profile {
            browser = browser.profile(dir);
        }
        browser = browser.careful(careful);
        if let Some(url) = start_url {
            browser = browser.start_url(url);
        }
        Self::update(slf, |b| Ok(b.browser(browser)))
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

    /// How much the model thinks before it answers: `"low"`, `"medium"` or `"high"`.
    fn thinking_effort<'py>(
        slf: PyRef<'py, Self>,
        effort: Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let effort: ThinkingEffort = from_py(&effort, "a thinking effort")?;
        Self::update(slf, |b| Ok(b.thinking_effort(effort)))
    }

    /// Constrain replies to the JSON schema `schema`.
    fn output_schema<'py>(
        slf: PyRef<'py, Self>,
        schema: Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let schema: Value = from_py(&schema, "a JSON schema")?;
        Self::update(slf, |b| Ok(b.output_schema(schema)))
    }

    /// Make the agent — an awaitable, as `AgentBuilder::build` is a future: it builds the
    /// model and reads the skills through the console.
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
pub fn item<T: serde::Serialize>(item: anyhow::Result<T>) -> PyResult<Py<PyAny>> {
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
    #[pyo3(signature = (spec, *, history = None, console = None, memory = None))]
    fn from_spec<'py>(
        py: Python<'py>,
        spec: Bound<'py, PyAny>,
        history: Option<Bound<'py, PyAny>>,
        console: Option<Bound<'py, PyAny>>,
        memory: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec: AgentSpec = from_py(&spec, "an agent spec")?;
        let mut state = AgentState::new();
        if let Some(history) = history {
            state = state.with_history(from_py::<Vec<Message>>(&history, "a list of messages")?);
        }
        if let Some(console) = console {
            state = state.with_console_slot(console::attach(&console)?);
        }
        if let Some(memfile) = memory {
            state = state.with_memory(Memory::new(memfile));
        }
        future_into_py(py, async move {
            let agent = Agent::try_with_state(spec, state)
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
        to_py(py, &agent.history())
    }

    /// Connect the agent's browser now rather than on its first use. This is when the browser asks,
    /// so call it while the person is at their screen; a refusal raises. A no-op once
    /// connected. Awaitable; raises during a turn.
    fn start_browser<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut guard = self
            .agent
            .clone()
            .try_lock_owned()
            .map_err(|_| AiloyError::new_err("this agent is in the middle of a turn"))?;
        future_into_py(py, async move {
            let agent = guard.0.as_mut().ok_or_else(closed)?;
            agent.start_browser().await.map_err(error::anyhow)
        })
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
    m.add_class::<PyAgentBuilder>()?;
    m.add_class::<PyAgent>()?;
    m.add_class::<PyAgentRun>()?;
    Ok(())
}
