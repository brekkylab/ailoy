use futures::StreamExt;
use pyo3::{
    IntoPyObjectExt, PyClass,
    exceptions::{PyNotImplementedError, PyRuntimeError, PyStopAsyncIteration, PyStopIteration},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::derive::*;
use tokio::runtime::Runtime;

use crate::{
    ffi::py::{
        base::{PyWrapper, await_future},
        cache_progress::await_cache_result,
    },
    model::{
        LanguageModel, LocalLanguageModel, anthropic::AnthropicLanguageModel,
        gemini::GeminiLanguageModel, openai::OpenAILanguageModel, xai::XAILanguageModel,
    },
    value::{Message, MessageOutput, ToolDesc},
};

pub trait PyLanguageModelMethods<T: LanguageModel + 'static>:
    PyWrapper<Inner = T> + PyClass<BaseType = PyLanguageModel>
{
    fn _spawn(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDesc>>,
    ) -> PyResult<(
        &'static tokio::runtime::Runtime,
        async_channel::Receiver<Result<MessageOutput, String>>,
    )> {
        let model = self.into_inner()?;

        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.spawn(async move {
            let mut model = model;
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
#[pyclass(name = "LanguageModel", subclass)]
pub struct PyLanguageModel {}

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
    inner: LocalLanguageModel,
}

impl PyWrapper for PyLocalLanguageModel {
    type Inner = LocalLanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, (Self { inner }, PyLanguageModel {}))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyLanguageModelMethods<LocalLanguageModel> for PyLocalLanguageModel {}

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
            Python::attach(|py| Py::new(py, (PyLocalLanguageModel { inner }, PyLanguageModel {})))
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
    ) -> PyResult<Py<PyLocalLanguageModel>> {
        let inner = await_future(await_cache_result::<LocalLanguageModel>(
            model_name,
            progress_callback,
        ))?;
        Py::new(py, (PyLocalLanguageModel { inner }, PyLanguageModel {}))
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

    fn enable_reasoning(&self) {
        self.inner.enable_reasoning();
    }

    fn disable_reasoning(&self) {
        self.inner.disable_reasoning();
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
#[pyclass(name = "OpenAILanguageModel", extends = PyLanguageModel)]
pub struct PyOpenAILanguageModel {
    inner: OpenAILanguageModel,
}

impl PyWrapper for PyOpenAILanguageModel {
    type Inner = OpenAILanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, (Self { inner }, PyLanguageModel {}))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyLanguageModelMethods<OpenAILanguageModel> for PyOpenAILanguageModel {}

#[gen_stub_pymethods]
#[pymethods]
impl PyOpenAILanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(OpenAILanguageModel::new(model_name, api_key), py)
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
#[pyclass(name = "GeminiLanguageModel", extends = PyLanguageModel)]
pub struct PyGeminiLanguageModel {
    inner: GeminiLanguageModel,
}

impl PyWrapper for PyGeminiLanguageModel {
    type Inner = GeminiLanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, (Self { inner }, PyLanguageModel {}))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyLanguageModelMethods<GeminiLanguageModel> for PyGeminiLanguageModel {}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeminiLanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(GeminiLanguageModel::new(model_name, api_key), py)
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
#[pyclass(name = "AnthropicLanguageModel", extends = PyLanguageModel)]
pub struct PyAnthropicLanguageModel {
    inner: AnthropicLanguageModel,
}

impl PyWrapper for PyAnthropicLanguageModel {
    type Inner = AnthropicLanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, (Self { inner }, PyLanguageModel {}))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyLanguageModelMethods<AnthropicLanguageModel> for PyAnthropicLanguageModel {}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnthropicLanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(AnthropicLanguageModel::new(model_name, api_key), py)
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
#[pyclass(name = "XAILanguageModel", extends = PyLanguageModel)]
pub struct PyXAILanguageModel {
    inner: XAILanguageModel,
}

impl PyWrapper for PyXAILanguageModel {
    type Inner = XAILanguageModel;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, (Self { inner }, PyLanguageModel {}))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyLanguageModelMethods<XAILanguageModel> for PyXAILanguageModel {}

#[gen_stub_pymethods]
#[pymethods]
impl PyXAILanguageModel {
    #[new]
    fn __new__(py: Python<'_>, model_name: String, api_key: String) -> PyResult<Py<Self>> {
        Self::into_py_obj(XAILanguageModel::new(model_name, api_key), py)
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

impl<'py> IntoPyObject<'py> for Box<dyn LanguageModel> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_any = if let Some(model) = self.downcast_ref::<LocalLanguageModel>() {
            PyLocalLanguageModel::into_py_obj(model.clone(), py)?.into_py_any(py)
        } else if let Some(model) = self.downcast_ref::<OpenAILanguageModel>() {
            PyOpenAILanguageModel::into_py_obj(model.clone(), py)?.into_py_any(py)
        } else if let Some(model) = self.downcast_ref::<GeminiLanguageModel>() {
            PyGeminiLanguageModel::into_py_obj(model.clone(), py)?.into_py_any(py)
        } else if let Some(model) = self.downcast_ref::<AnthropicLanguageModel>() {
            PyAnthropicLanguageModel::into_py_obj(model.clone(), py)?.into_py_any(py)
        } else if let Some(model) = self.downcast_ref::<XAILanguageModel>() {
            PyXAILanguageModel::into_py_obj(model.clone(), py)?.into_py_any(py)
        } else {
            Err(PyRuntimeError::new_err("Failed to downcast LanguageModel"))
        }?;
        Ok(py_any.into_bound(py))
    }
}
