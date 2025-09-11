use std::sync::Arc;

use futures::StreamExt;
use pyo3::{
    Python,
    exceptions::{PyRuntimeError, PyStopAsyncIteration, PyStopIteration, PyTypeError},
    prelude::*,
    types::PyList,
};
use pyo3_stub_gen::derive::*;

use crate::{
    agent::Agent,
    ffi::py::{
        base::{PyWrapper, await_future},
        language_model::{
            PyAnthropicLanguageModel, PyGeminiLanguageModel, PyLanguageModel, PyLocalLanguageModel,
            PyOpenAILanguageModel, PyXAILanguageModel,
        },
        tool::ArcTool,
    },
    tool::Tool,
    value::MessageOutput,
};

#[gen_stub_pyclass]
#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Agent,
}

fn pylist_to_tools(tools: Py<PyList>) -> PyResult<Vec<Arc<dyn Tool>>> {
    Python::attach(|py| {
        tools
            .bind(py)
            .into_iter()
            .map(|tool| tool.extract::<ArcTool>().map(|arc| arc.0))
            .collect::<PyResult<Vec<Arc<dyn Tool>>>>()
    })
}

impl PyAgent {
    fn _spawn(
        &self,
        message: String,
    ) -> PyResult<(
        &'static tokio::runtime::Runtime,
        async_channel::Receiver<Result<MessageOutput, String>>,
    )> {
        let mut inner = self.inner.clone();

        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();

        rt.spawn(async move {
            let mut strm = inner.run(message).boxed();

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
        lm: Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "list[Tool]"))] tools: Option<Py<PyList>>,
    ) -> PyResult<Py<Self>> {
        if !lm.is_instance_of::<PyLanguageModel>() {
            return Err(PyTypeError::new_err(
                "lm must be a subclass of LanguageModel",
            ));
        }

        let tools = tools.map_or_else(|| Ok(vec![]), |tools| pylist_to_tools(tools))?;

        let agent: Agent = if let Ok(lm) = lm.downcast::<PyLocalLanguageModel>() {
            let model = lm.borrow_mut().into_inner()?;
            Agent::new(model, tools)
        } else if let Ok(lm) = lm.downcast::<PyOpenAILanguageModel>() {
            let model = lm.borrow_mut().into_inner()?;
            Agent::new(model, tools)
        } else if let Ok(lm) = lm.downcast::<PyGeminiLanguageModel>() {
            let model = lm.borrow_mut().into_inner()?;
            Agent::new(model, tools)
        } else if let Ok(lm) = lm.downcast::<PyAnthropicLanguageModel>() {
            let model = lm.borrow_mut().into_inner()?;
            Agent::new(model, tools)
        } else if let Ok(lm) = lm.downcast::<PyXAILanguageModel>() {
            let model = lm.borrow_mut().into_inner()?;
            Agent::new(model, tools)
        } else {
            return Err(PyTypeError::new_err("Unknown language model provided"));
        };

        Py::new(py, Self { inner: agent })
    }

    #[gen_stub(override_return_type(type_repr = "LanguageModel"))]
    #[getter]
    fn lm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let lm_ref = self.inner.get_lm();
        let obj = lm_ref.clone().into_pyobject(py)?;
        Ok(obj.unbind())
    }

    #[gen_stub(override_return_type(type_repr = "list[Tool]"))]
    #[getter]
    fn tools(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let tools = self
            .inner
            .get_tools()
            .into_iter()
            .map(|t| ArcTool(t).into_pyobject(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, tools)?.unbind())
    }

    fn add_tools(
        &mut self,
        #[gen_stub(override_type(type_repr = "list[Tool]"))] tools: Py<PyList>,
    ) -> PyResult<()> {
        let tools = pylist_to_tools(tools)?;
        await_future(self.inner.add_tools(tools))
    }

    fn add_tool(
        &mut self,
        #[gen_stub(override_type(type_repr = "Tool"))] tool: Py<PyAny>,
    ) -> PyResult<()> {
        let tool = Python::attach(|py| tool.extract::<ArcTool>(py).map(|arc| arc.0))?;
        await_future(self.inner.add_tool(tool))
    }

    fn remove_tools(&mut self, tool_names: Vec<String>) -> PyResult<()> {
        await_future(self.inner.remove_tools(tool_names))
    }

    fn remove_tool(&mut self, tool_name: String) -> PyResult<()> {
        await_future(self.inner.remove_tool(tool_name))
    }

    fn run(&self, message: String) -> PyResult<PyAgentRunIterator> {
        let (_, rx) = self._spawn(message)?;
        Ok(PyAgentRunIterator { rx })
    }

    fn run_sync(&mut self, message: String) -> PyResult<PyAgentRunSyncIterator> {
        let (rt, rx) = self._spawn(message)?;
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

    #[gen_stub(override_return_type(type_repr="typing.Awaitable[MessageOutput]", imports=("typing")))]
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
