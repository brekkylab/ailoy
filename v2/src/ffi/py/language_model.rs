use std::any::TypeId;

use futures::StreamExt;
use pyo3::{
    IntoPyObjectExt, PyClass,
    exceptions::{
        PyNotImplementedError, PyRuntimeError, PyStopAsyncIteration, PyStopIteration, PyTypeError,
    },
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::derive::*;
use tokio::runtime::Runtime;

use crate::{
    ffi::py::{base::await_future, cache_progress::await_cache_result},
    model::{
        ArcMutexLanguageModel, LocalLanguageModel, anthropic::AnthropicLanguageModel,
        gemini::GeminiLanguageModel, openai::OpenAILanguageModel, xai::XAILanguageModel,
    },
    value::{Message, MessageOutput, ToolDesc},
};

pub trait PyLanguageModelMethods: PyClass<BaseType = PyBaseLanguageModel> {
    fn from_inner(inner: ArcMutexLanguageModel) -> Self;

    fn into_inner(&self) -> ArcMutexLanguageModel;

    fn into_py_obj(py: Python<'_>, inner: ArcMutexLanguageModel) -> PyResult<Py<Self>> {
        Py::new(py, (Self::from_inner(inner), PyBaseLanguageModel {}))
    }

    fn _spawn(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<(
        &'static tokio::runtime::Runtime,
        async_channel::Receiver<Result<MessageOutput, String>>,
    )> {
        let model = self.into_inner().model;

        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.spawn(async move {
            let mut model = model.lock().await;
            let mut stream = model.run(messages, tools.unwrap_or(vec![])).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });

        Ok((rt, rx))
    }

    fn _run(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        let (_, rx) = self._spawn(messages, tools)?;
        Ok(PyLanguageModelRunIterator { rx })
    }

    fn _run_sync(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        let (rt, rx) = self._spawn(messages, tools)?;
        Ok(PyLanguageModelRunSyncIterator { rt, rx })
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "BaseLanguageModel", subclass)]
pub struct PyBaseLanguageModel {}

#[gen_stub_pymethods]
#[pymethods]
impl PyBaseLanguageModel {
    #[allow(unused_variables)]
    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        Err(PyNotImplementedError::new_err(
            "Subclasses must implement 'run'",
        ))
    }

    #[allow(unused_variables)]
    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        Err(PyNotImplementedError::new_err(
            "Subclasses must implement 'run_sync'",
        ))
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "LocalLanguageModel", extends = PyBaseLanguageModel)]
pub struct PyLocalLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl PyLanguageModelMethods for PyLocalLanguageModel {
    fn from_inner(inner: ArcMutexLanguageModel) -> Self {
        Self { inner }
    }

    fn into_inner(&self) -> ArcMutexLanguageModel {
        self.inner.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLocalLanguageModel {
    #[classmethod]
    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[LocalLanguageModel]"))]
    #[pyo3(signature = (model_name, progress_callback = None))]
    fn create<'a>(
        _cls: &Bound<'a, PyType>,
        py: Python<'a>,
        model_name: String,
        #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let fut = async move {
            let inner =
                await_cache_result::<LocalLanguageModel>(model_name, progress_callback).await?;
            Python::attach(|py| Self::into_py_obj(py, ArcMutexLanguageModel::new(inner)))
        };
        pyo3_async_runtimes::tokio::future_into_py(py, fut)
    }

    #[classmethod]
    #[pyo3(signature = (model_name, progress_callback = None))]
    fn create_sync(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model_name: String,
        #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<Py<Self>> {
        let inner = await_future(await_cache_result::<LocalLanguageModel>(
            model_name,
            progress_callback,
        ))?;
        Py::new(
            py,
            (
                Self::from_inner(ArcMutexLanguageModel::new(inner)),
                PyBaseLanguageModel {},
            ),
        )
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self._run(messages, tools)
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self._run_sync(messages, tools)
    }

    fn enable_reasoning(&self) -> PyResult<()> {
        await_future(async move {
            let arc_model = self
                .inner
                .into_inner::<LocalLanguageModel>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let model = arc_model.lock().await;
            model.enable_reasoning();
            Ok::<(), PyErr>(())
        })
    }

    fn disable_reasoning(&self) -> PyResult<()> {
        await_future(async move {
            let arc_model = self
                .inner
                .into_inner::<LocalLanguageModel>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let model = arc_model.lock().await;
            model.disable_reasoning();
            Ok::<(), PyErr>(())
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "LanguageModelRunIterator")]
pub struct PyLanguageModelRunIterator {
    rx: async_channel::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLanguageModelRunIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MessageOutput]"))]
    fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let fut = async move {
            match rx.recv().await {
                Ok(res) => res.map_err(|e| PyRuntimeError::new_err(e.to_string())),
                Err(_) => Err(PyStopAsyncIteration::new_err(())),
            }
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "LanguageModelRunSyncIterator")]
pub struct PyLanguageModelRunSyncIterator {
    rt: &'static Runtime,
    rx: async_channel::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLanguageModelRunSyncIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<MessageOutput> {
        let item = py.detach(|| self.rt.block_on(self.rx.recv()));
        match item {
            Ok(res) => res.map_err(|e| PyRuntimeError::new_err(e.to_string())),
            Err(_) => Err(PyStopIteration::new_err(())),
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "OpenAILanguageModel", extends = PyBaseLanguageModel)]
pub struct PyOpenAILanguageModel {
    inner: ArcMutexLanguageModel,
}

impl PyLanguageModelMethods for PyOpenAILanguageModel {
    fn from_inner(inner: ArcMutexLanguageModel) -> Self {
        Self { inner }
    }

    fn into_inner(&self) -> ArcMutexLanguageModel {
        self.inner.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAILanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Py::new(
            py,
            (
                Self::from_inner(ArcMutexLanguageModel::new(OpenAILanguageModel::new(
                    model_name, api_key,
                ))),
                PyBaseLanguageModel {},
            ),
        )
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self._run(messages, tools)
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self._run_sync(messages, tools)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "GeminiLanguageModel", extends = PyBaseLanguageModel)]
pub struct PyGeminiLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl PyLanguageModelMethods for PyGeminiLanguageModel {
    fn from_inner(inner: ArcMutexLanguageModel) -> Self {
        Self { inner }
    }

    fn into_inner(&self) -> ArcMutexLanguageModel {
        self.inner.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeminiLanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Py::new(
            py,
            (
                Self::from_inner(ArcMutexLanguageModel::new(GeminiLanguageModel::new(
                    model_name, api_key,
                ))),
                PyBaseLanguageModel {},
            ),
        )
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self._run(messages, tools)
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self._run_sync(messages, tools)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "AnthropicLanguageModel", extends = PyBaseLanguageModel)]
pub struct PyAnthropicLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl PyLanguageModelMethods for PyAnthropicLanguageModel {
    fn from_inner(inner: ArcMutexLanguageModel) -> Self {
        Self { inner }
    }

    fn into_inner(&self) -> ArcMutexLanguageModel {
        self.inner.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnthropicLanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Py::new(
            py,
            (
                Self::from_inner(ArcMutexLanguageModel::new(AnthropicLanguageModel::new(
                    model_name, api_key,
                ))),
                PyBaseLanguageModel {},
            ),
        )
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self._run(messages, tools)
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self._run_sync(messages, tools)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "XAILanguageModel", extends = PyBaseLanguageModel)]
pub struct PyXAILanguageModel {
    inner: ArcMutexLanguageModel,
}

impl PyLanguageModelMethods for PyXAILanguageModel {
    fn from_inner(inner: ArcMutexLanguageModel) -> Self {
        Self { inner }
    }

    fn into_inner(&self) -> ArcMutexLanguageModel {
        self.inner.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyXAILanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Py::new(
            py,
            (
                Self::from_inner(ArcMutexLanguageModel::new(XAILanguageModel::new(
                    model_name, api_key,
                ))),
                PyBaseLanguageModel {},
            ),
        )
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self._run(messages, tools)
    }

    #[pyo3(signature = (messages, tools = None))]
    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self._run_sync(messages, tools)
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for ArcMutexLanguageModel {
    type Error = PyErr;

    fn try_from(any: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(lm) = any.downcast::<PyLocalLanguageModel>() {
            Ok(lm.borrow().into_inner())
        } else if let Ok(lm) = any.downcast::<PyOpenAILanguageModel>() {
            Ok(lm.borrow().into_inner())
        } else if let Ok(lm) = any.downcast::<PyGeminiLanguageModel>() {
            Ok(lm.borrow().into_inner())
        } else if let Ok(lm) = any.downcast::<PyAnthropicLanguageModel>() {
            Ok(lm.borrow().into_inner())
        } else if let Ok(lm) = any.downcast::<PyXAILanguageModel>() {
            Ok(lm.borrow().into_inner())
        } else {
            Err(PyTypeError::new_err("Unknown language model provided"))
        }
    }
}

impl<'py> IntoPyObject<'py> for ArcMutexLanguageModel {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_any = if self.type_id == TypeId::of::<LocalLanguageModel>() {
            PyLocalLanguageModel::into_py_obj(py, self)?.into_py_any(py)
        } else if self.type_id == TypeId::of::<OpenAILanguageModel>() {
            PyOpenAILanguageModel::into_py_obj(py, self)?.into_py_any(py)
        } else if self.type_id == TypeId::of::<GeminiLanguageModel>() {
            PyGeminiLanguageModel::into_py_obj(py, self)?.into_py_any(py)
        } else if self.type_id == TypeId::of::<AnthropicLanguageModel>() {
            PyAnthropicLanguageModel::into_py_obj(py, self)?.into_py_any(py)
        } else if self.type_id == TypeId::of::<XAILanguageModel>() {
            PyXAILanguageModel::into_py_obj(py, self)?.into_py_any(py)
        } else {
            Err(PyRuntimeError::new_err(
                "Failed to downcast BaseLanguageModel",
            ))
        }?;
        Ok(py_any.into_bound(py))
    }
}
