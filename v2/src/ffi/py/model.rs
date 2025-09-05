use futures::StreamExt;
use pyo3::{
    PyClass,
    exceptions::{PyNotImplementedError, PyRuntimeError, PyStopAsyncIteration, PyStopIteration},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::derive::*;
use tokio::runtime::Runtime;

use crate::{
    ffi::py::{
        base::PyWrapper,
        cache_progress::{
            PyCacheProgressIterator, PyCacheProgressSyncIterator, create_cache_progress_iterator,
            create_cache_progress_sync_iterator,
        },
    },
    model::{
        LanguageModel, LocalLanguageModel, anthropic::AnthropicLanguageModel,
        gemini::GeminiLanguageModel, openai::OpenAILanguageModel, xai::XAILanguageModel,
    },
    value::{Message, MessageOutput, ToolDesc},
};

#[gen_stub_pyclass]
#[pyclass(name = "LanguageModel", subclass)]
pub struct PyLanguageModel {}

trait PyLanguageModelMethods: PyClass {
    type Inner: LanguageModel;

    fn inner(&mut self) -> Result<Self::Inner, PyErr>;

    fn run(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        let model = self.inner()?;

        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let mut model = model;
            let mut strm = model.run(messages, tools).boxed();

            while let Some(item) = strm.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });
        Ok(PyLanguageModelRunIterator { rx })
    }

    fn run_sync(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        let model = self.inner()?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<MessageOutput, String>>(16);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        rt.spawn(async move {
            let mut model = model;
            let mut stream = model.run(messages, tools).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        Ok(PyLanguageModelRunSyncIterator { rt, rx })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLanguageModel {
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
#[pyclass(name = "LocalLanguageModel", extends = PyLanguageModel)]
pub struct PyLocalLanguageModel {
    inner: Option<LocalLanguageModel>,
}

impl PyWrapper for PyLocalLanguageModel {
    type Inner = LocalLanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyLanguageModel {};
        let child = Self { inner: Some(inner) };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }
}

impl PyLanguageModelMethods for PyLocalLanguageModel {
    type Inner = LocalLanguageModel;

    fn inner(&mut self) -> Result<Self::Inner, PyErr> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Model already consumed"))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLocalLanguageModel {
    // Async version of creation
    #[classmethod]
    #[gen_stub(override_return_type(type_repr="CacheProgressIterator[LocalLanguageModel]", imports=("typing")))]
    pub fn create(
        _cls: &Bound<'_, PyType>,
        py: Python,
        model_name: &str,
    ) -> PyResult<Py<PyCacheProgressIterator>> {
        create_cache_progress_iterator::<PyLocalLanguageModel>(model_name.to_string(), py)
    }

    // Sync version of creation
    #[classmethod]
    #[gen_stub(override_return_type(type_repr="CacheProgressSyncIterator[LocalLanguageModel]", imports=("typing")))]
    pub fn create_sync(
        _cls: &Bound<'_, PyType>,
        py: Python,
        model_name: &str,
    ) -> PyResult<Py<PyCacheProgressSyncIterator>> {
        create_cache_progress_sync_iterator::<PyLocalLanguageModel>(model_name, py)
    }

    #[pyo3(name = "run")]
    fn run_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self.run(messages, tools)
    }

    #[pyo3(name = "run_sync")]
    fn run_sync_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self.run_sync(messages, tools)
    }

    fn enable_reasoning(&self) {
        match &self.inner {
            Some(inner) => {
                inner.enable_reasoning();
            }
            None => {}
        }
    }

    fn disable_reasoning(&self) {
        match &self.inner {
            Some(inner) => {
                inner.disable_reasoning();
            }
            None => {}
        }
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
#[pyclass(unsendable, name = "LanguageModelRunSyncIterator")]
pub struct PyLanguageModelRunSyncIterator {
    rt: Runtime,
    rx: tokio::sync::mpsc::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLanguageModelRunSyncIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<MessageOutput> {
        let item = py.allow_threads(|| self.rt.block_on(self.rx.recv()));
        match item {
            Some(Ok(evt)) => Ok(evt),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Err(PyStopIteration::new_err(())), // StopIteration
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "OpenAILanguageModel", extends = PyLanguageModel)]
pub struct PyOpenAILanguageModel {
    inner: OpenAILanguageModel,
}

impl PyWrapper for PyOpenAILanguageModel {
    type Inner = OpenAILanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyLanguageModel {};
        let child = Self { inner: inner };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }
}

impl PyLanguageModelMethods for PyOpenAILanguageModel {
    type Inner = OpenAILanguageModel;

    fn inner(&mut self) -> Result<Self::Inner, PyErr> {
        Ok(self.inner.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAILanguageModel {
    #[new]
    pub fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(OpenAILanguageModel::new(model_name, api_key), py)
    }

    #[pyo3(name = "run")]
    fn run_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self.run(messages, tools)
    }

    #[pyo3(name = "run_sync")]
    fn run_sync_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self.run_sync(messages, tools)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "GeminiLanguageModel", extends = PyLanguageModel)]
pub struct PyGeminiLanguageModel {
    inner: GeminiLanguageModel,
}

impl PyWrapper for PyGeminiLanguageModel {
    type Inner = GeminiLanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyLanguageModel {};
        let child = Self { inner: inner };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }
}

impl PyLanguageModelMethods for PyGeminiLanguageModel {
    type Inner = GeminiLanguageModel;

    fn inner(&mut self) -> Result<Self::Inner, PyErr> {
        Ok(self.inner.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeminiLanguageModel {
    #[new]
    pub fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(GeminiLanguageModel::new(model_name, api_key), py)
    }

    #[pyo3(name = "run")]
    fn run_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self.run(messages, tools)
    }

    #[pyo3(name = "run_sync")]
    fn run_sync_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self.run_sync(messages, tools)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "AnthropicLanguageModel", extends = PyLanguageModel)]
pub struct PyAnthropicLanguageModel {
    inner: AnthropicLanguageModel,
}

impl PyWrapper for PyAnthropicLanguageModel {
    type Inner = AnthropicLanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyLanguageModel {};
        let child = Self { inner: inner };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }
}

impl PyLanguageModelMethods for PyAnthropicLanguageModel {
    type Inner = AnthropicLanguageModel;

    fn inner(&mut self) -> Result<Self::Inner, PyErr> {
        Ok(self.inner.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnthropicLanguageModel {
    #[new]
    pub fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(AnthropicLanguageModel::new(model_name, api_key), py)
    }

    #[pyo3(name = "run")]
    fn run_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self.run(messages, tools)
    }

    #[pyo3(name = "run_sync")]
    fn run_sync_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self.run_sync(messages, tools)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "XAILanguageModel", extends = PyLanguageModel)]
pub struct PyXAILanguageModel {
    inner: XAILanguageModel,
}

impl PyWrapper for PyXAILanguageModel {
    type Inner = XAILanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyLanguageModel {};
        let child = Self { inner: inner };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }
}

impl PyLanguageModelMethods for PyXAILanguageModel {
    type Inner = XAILanguageModel;

    fn inner(&mut self) -> Result<Self::Inner, PyErr> {
        Ok(self.inner.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyXAILanguageModel {
    #[new]
    pub fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(XAILanguageModel::new(model_name, api_key), py)
    }

    #[pyo3(name = "run")]
    fn run_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        self.run(messages, tools)
    }

    #[pyo3(name = "run_sync")]
    fn run_sync_(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        self.run_sync(messages, tools)
    }
}
