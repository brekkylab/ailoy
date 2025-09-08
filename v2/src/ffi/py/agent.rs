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
        base::PyWrapper,
        model::{
            PyAnthropicLanguageModel, PyGeminiLanguageModel, PyLanguageModel, PyLocalLanguageModel,
            PyOpenAILanguageModel, PyXAILanguageModel,
        },
        tool::{
            PyBuiltinTool, PyMCPTool, PyToolMethods, PythonAsyncFunctionTool, PythonFunctionTool,
        },
    },
    model::{
        LocalLanguageModel, anthropic::AnthropicLanguageModel, gemini::GeminiLanguageModel,
        openai::OpenAILanguageModel, xai::XAILanguageModel,
    },
    tool::Tool,
    value::MessageOutput,
};

#[gen_stub_pyclass]
#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Agent,
}

impl PyWrapper for PyAgent {
    type Inner = Agent;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, Self { inner: inner })
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgent {
    #[new]
    pub fn __new__(
        py: Python<'_>,
        lm: Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.List[Tool]"))] tools: Bound<'_, PyList>,
    ) -> PyResult<Py<Self>> {
        if !lm.is_instance_of::<PyLanguageModel>() {
            return Err(PyTypeError::new_err(
                "lm must be a subclass of LanguageModel",
            ));
        }

        let tools = tools
            .into_iter()
            .map(|tool| {
                if let Ok(tool) = tool.downcast::<PyBuiltinTool>() {
                    Ok(Arc::new(tool.as_unbound().borrow(py).inner().clone()) as Arc<dyn Tool>)
                } else if let Ok(tool) = tool.downcast::<PyMCPTool>() {
                    Ok(Arc::new(tool.as_unbound().borrow(py).inner().clone()) as Arc<dyn Tool>)
                } else if let Ok(tool) = tool.downcast::<PythonFunctionTool>() {
                    Ok(Arc::new(tool.as_unbound().borrow(py).clone()) as Arc<dyn Tool>)
                } else if let Ok(tool) = tool.downcast::<PythonAsyncFunctionTool>() {
                    Ok(Arc::new(tool.as_unbound().borrow(py).clone()) as Arc<dyn Tool>)
                } else {
                    return Err(PyTypeError::new_err("Unknown tool provided"));
                }
            })
            .collect::<PyResult<Vec<Arc<dyn Tool>>>>()?;

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

    #[gen_stub(override_return_type(type_repr="LanguageModel", imports=()))]
    #[getter]
    pub fn lm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.into_inner()?;
        let lm_ref = inner.get_lm();

        if let Some(model) = lm_ref.downcast_ref::<LocalLanguageModel>() {
            let py_obj = PyLocalLanguageModel::into_py_obj(model.clone(), py)?;
            Ok(Py::new(py, py_obj)?.into())
        } else if let Some(model) = lm_ref.downcast_ref::<OpenAILanguageModel>() {
            let py_obj = PyOpenAILanguageModel::into_py_obj(model.clone(), py)?;
            Ok(Py::new(py, py_obj)?.into())
        } else if let Some(model) = lm_ref.downcast_ref::<GeminiLanguageModel>() {
            let py_obj = PyGeminiLanguageModel::into_py_obj(model.clone(), py)?;
            Ok(Py::new(py, py_obj)?.into())
        } else if let Some(model) = lm_ref.downcast_ref::<AnthropicLanguageModel>() {
            let py_obj = PyAnthropicLanguageModel::into_py_obj(model.clone(), py)?;
            Ok(Py::new(py, py_obj)?.into())
        } else if let Some(model) = lm_ref.downcast_ref::<XAILanguageModel>() {
            let py_obj = PyXAILanguageModel::into_py_obj(model.clone(), py)?;
            Ok(Py::new(py, py_obj)?.into())
        } else {
            Err(PyRuntimeError::new_err("Failed to downcast lm"))
        }
    }

    fn run(&mut self, message: String) -> PyResult<PyAgentRunIterator> {
        let mut inner = self.into_inner()?;

        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let mut strm = inner.run(message).boxed();

            while let Some(item) = strm.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });
        Ok(PyAgentRunIterator { rx })
    }

    fn run_sync(&mut self, message: String) -> PyResult<PyAgentRunSyncIterator> {
        let mut inner = self.into_inner()?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<MessageOutput, String>>(16);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        rt.spawn(async move {
            let mut stream = inner.run(message).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

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
                Ok(Ok(evt)) => Ok(evt),
                Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
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
    rt: tokio::runtime::Runtime,
    rx: tokio::sync::mpsc::Receiver<Result<MessageOutput, String>>,
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
            Some(Ok(evt)) => Ok(evt),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Err(PyStopIteration::new_err(())), // StopIteration
        }
    }
}
