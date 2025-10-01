use std::sync::Arc;

use futures::{StreamExt, lock::Mutex};
use pyo3::{
    Python,
    exceptions::{PyRuntimeError, PyStopAsyncIteration, PyStopIteration, PyTypeError},
    prelude::*,
    types::{PyList, PyString},
};
use pyo3_stub_gen::derive::*;

use crate::{
    agent::Agent,
    ffi::py::{base::await_future, language_model::PyBaseLanguageModel},
    model::ArcMutexLanguageModel,
    tool::{ArcTool, Tool},
    value::{MessageOutput, Part},
};

#[gen_stub_pyclass]
#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Arc<Mutex<Agent>>,
}

fn pylist_to_tools(tools: Py<PyList>) -> PyResult<Vec<Arc<dyn Tool>>> {
    Python::attach(|py| {
        tools
            .bind(py)
            .into_iter()
            .map(|tool| tool.extract::<ArcTool>().map(|arc| arc.inner))
            .collect::<PyResult<Vec<Arc<dyn Tool>>>>()
    })
}

impl PyAgent {
    fn _spawn(
        &self,
        py: Python<'_>,
        contents: Py<PyAny>,
    ) -> PyResult<(
        &'static tokio::runtime::Runtime,
        async_channel::Receiver<Result<MessageOutput, String>>,
    )> {
        let inner = self.inner.clone();

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();

        let contents_bound = contents.bind(py);
        let parts = if let Ok(list) = contents_bound.downcast::<PyList>() {
            list.into_iter()
                .map(|item| {
                    if let Ok(item) = item.downcast::<Part>() {
                        Ok(item.as_unbound().borrow(py).clone())
                    } else {
                        // If the list item is not a type of part, it's just converted to a text type part.
                        let text = item.to_string();
                        Ok(Part::new_text(text))
                    }
                })
                .collect::<PyResult<Vec<Part>>>()
        } else if let Ok(str) = contents_bound.downcast::<PyString>() {
            let text = str.to_string();
            Ok(vec![Part::new_text(text)])
        } else {
            Err(PyTypeError::new_err(
                "contents should be either a str or a list of Part",
            ))
        }?;

        rt.spawn(async move {
            let mut inner = inner.lock().await;
            let mut strm = inner.run(parts).boxed();

            while let Some(item) = strm.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });

        Ok((rt, rx))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgent {
    #[new]
    #[pyo3(signature = (lm, tools = None))]
    fn __new__(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "BaseLanguageModel"))] lm: Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "list[BaseTool]"))] tools: Option<Py<PyList>>,
    ) -> PyResult<Py<Self>> {
        if !lm.is_instance_of::<PyBaseLanguageModel>() {
            return Err(PyTypeError::new_err(
                "lm must be a subclass of BaseLanguageModel",
            ));
        }

        let tools = tools.map_or_else(|| Ok(vec![]), |tools| pylist_to_tools(tools))?;
        let model: ArcMutexLanguageModel = lm.try_into()?;
        let agent = Agent::new_from_arc(model, tools);

        Py::new(
            py,
            Self {
                inner: Arc::new(Mutex::new(agent)),
            },
        )
    }

    #[gen_stub(override_return_type(type_repr = "BaseLanguageModel"))]
    #[getter]
    fn lm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let lm = await_future(async { Ok::<_, PyErr>(self.inner.lock().await.get_lm()) })?;
        let obj = lm.into_pyobject(py)?;
        Ok(obj.unbind())
    }

    #[gen_stub(override_return_type(type_repr = "list[BaseTool]"))]
    #[getter]
    fn tools(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let tools = await_future(async {
            self.inner
                .lock()
                .await
                .get_tools()
                .into_iter()
                .map(|t| ArcTool::new_from_arc_any(t).into_pyobject(py))
                .collect::<PyResult<Vec<_>>>()
        })?;
        Ok(PyList::new(py, tools)?.unbind())
    }

    fn add_tools(
        &mut self,
        #[gen_stub(override_type(type_repr = "list[BaseTool]"))] tools: Py<PyList>,
    ) -> PyResult<()> {
        let tools = pylist_to_tools(tools)?;
        await_future(async { self.inner.lock().await.add_tools(tools).await })
    }

    fn add_tool(
        &mut self,
        #[gen_stub(override_type(type_repr = "BaseTool"))] tool: Py<PyAny>,
    ) -> PyResult<()> {
        let tool = Python::attach(|py| tool.extract::<ArcTool>(py).map(|arc| arc.inner))?;
        await_future(async { self.inner.lock().await.add_tool(tool).await })
    }

    fn remove_tools(&mut self, tool_names: Vec<String>) -> PyResult<()> {
        await_future(async { self.inner.lock().await.remove_tools(tool_names).await })
    }

    fn remove_tool(&mut self, tool_name: String) -> PyResult<()> {
        await_future(async { self.inner.lock().await.remove_tool(tool_name).await })
    }

    fn run(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "str | list[Part]"))] contents: Py<PyAny>,
    ) -> PyResult<PyAgentRunIterator> {
        let (_, rx) = self._spawn(py, contents)?;
        Ok(PyAgentRunIterator { rx })
    }

    fn run_sync(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "str | list[Part]"))] contents: Py<PyAny>,
    ) -> PyResult<PyAgentRunSyncIterator> {
        let (rt, rx) = self._spawn(py, contents)?;
        Ok(PyAgentRunSyncIterator { rt, rx })
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "AgentRunIterator")]
pub struct PyAgentRunIterator {
    rx: async_channel::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentRunIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MessageOutput]"))]
    fn __anext__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let fut = async move {
            match rx.recv().await {
                Ok(res) => res.map_err(|e| PyRuntimeError::new_err(e)),
                Err(_) => Err(PyStopAsyncIteration::new_err(())),
            }
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "AgentRunSyncIterator")]
pub struct PyAgentRunSyncIterator {
    rt: &'static tokio::runtime::Runtime,
    rx: async_channel::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentRunSyncIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<MessageOutput> {
        let item = py.detach(|| self.rt.block_on(self.rx.recv()));
        match item {
            Ok(res) => res.map_err(|e| PyRuntimeError::new_err(e)),
            Err(_) => Err(PyStopIteration::new_err(())),
        }
    }
}
