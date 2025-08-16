use std::sync::Arc;

use futures::{StreamExt as _, stream::BoxStream};
use pyo3::{
    exceptions::{PyRuntimeError, PyStopAsyncIteration},
    prelude::*,
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
        value::{PyMessage, PyMessageDelta},
    },
    model::{LanguageModel, LocalLanguageModel},
    value::MessageDelta,
};

#[gen_stub_pyclass]
#[pyclass(name = "LocalLanguageModel")]
pub struct PyLocalLanguageModel {
    inner: Arc<LocalLanguageModel>,
}

impl PyWrapper for PyLocalLanguageModel {
    type Inner = LocalLanguageModel;

    fn from_inner(inner: Self::Inner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLocalLanguageModel {
    // Async version of creation
    #[staticmethod]
    pub fn create(model_name: &str) -> PyCacheProgressIterator {
        create_cache_progress_iterator::<PyLocalLanguageModel>(model_name.to_string())
    }

    // Sync version of creation
    #[staticmethod]
    pub fn create_sync(model_name: &str) -> PyResult<PyCacheProgressSyncIterator> {
        create_cache_progress_sync_iterator::<PyLocalLanguageModel>(model_name)
    }

    pub fn run(&mut self, messages: Vec<PyMessage>) -> PyResult<PyAgentRunIterator> {
        let (tx, rx) = async_channel::unbounded::<Result<PyMessageDelta, String>>();
        let messages = messages.into_iter().map(|m| m.inner).collect::<Vec<_>>();
        let mut strm: BoxStream<'static, Result<MessageDelta, String>> =
            Box::pin(self.inner.clone().run(Vec::new(), messages));

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            while let Some(item) = strm.next().await {
                let out = item.and_then(|v| Ok(PyMessageDelta::from_inner(v)));
                if tx.send(out).await.is_err() {
                    break; // Exit if consumer vanished
                }
            }
        });
        Ok(PyAgentRunIterator { rx })
    }

    pub fn run_sync(&mut self, messages: Vec<PyMessage>) -> PyResult<PyAgentRunSyncIterator> {
        let messages = messages.into_iter().map(|m| m.inner).collect::<Vec<_>>();

        let lm = self.inner.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let strm = lm
            .run(Vec::new(), messages)
            .then(|item| async move { item.map(|v| PyMessageDelta::from_inner(v)) })
            .boxed();

        Ok(PyAgentRunSyncIterator { rt, strm })
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "AgentRunIterator")]
pub struct PyAgentRunIterator {
    rx: async_channel::Receiver<Result<PyMessageDelta, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentRunIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let fut = async move {
            match rx.recv().await {
                Ok(Ok(evt)) => Ok(evt),
                Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
                Err(_) => Err(PyStopAsyncIteration::new_err(())),
            }
        };
        pyo3_async_runtimes::tokio::future_into_py(py, fut).map(|a| a.unbind())
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "AgentRunSyncIterator")]
pub struct PyAgentRunSyncIterator {
    rt: Runtime,

    strm: BoxStream<'static, Result<PyMessageDelta, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentRunSyncIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyMessageDelta>> {
        let item = py.allow_threads(|| self.rt.block_on(async { self.strm.next().await }));
        match item {
            Some(Ok(evt)) => Ok(Some(evt)),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Ok(None), // StopIteration
        }
    }
}
