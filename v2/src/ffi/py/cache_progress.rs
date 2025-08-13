use futures::{StreamExt as _, stream::BoxStream};
use pyo3::{
    exceptions::{PyRuntimeError, PyStopAsyncIteration},
    prelude::*,
};
use tokio::runtime::Runtime;

use crate::{
    cache::{Cache, CacheProgress, TryFromCache},
    ffi::py::base::PyWrapper,
};

#[pyclass(name = "CacheProgress")]
pub struct PyCacheProgress {
    #[pyo3(get)]
    pub comment: String,

    #[pyo3(get)]
    pub current: usize,

    #[pyo3(get)]
    pub total: usize,

    #[pyo3(get)]
    pub result: Option<Py<PyAny>>,
}

#[pyclass(unsendable, name = "CacheProgressSyncIterator")]
pub struct PyCacheProgressSyncIterator {
    rt: Runtime,

    strm: BoxStream<'static, Result<PyCacheProgress, String>>,
}

#[pymethods]
impl PyCacheProgressSyncIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyCacheProgress>> {
        let item = py.allow_threads(|| self.rt.block_on(async { self.strm.next().await }));
        match item {
            Some(Ok(evt)) => Ok(Some(evt)),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Ok(None), // StopIteration
        }
    }
}

pub fn create_cache_progress_sync_iterator<T>(
    model_key: impl Into<String>,
) -> PyResult<PyCacheProgressSyncIterator>
where
    T: PyWrapper,
    T::Inner: TryFromCache,
{
    // attach current-thread runtime
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Rust stream
    let strm: BoxStream<'static, Result<CacheProgress<T::Inner>, String>> =
        Box::pin(Cache::new().try_create::<T::Inner>(model_key));

    // Map CacheProgress<T> -> PyCacheProgressIterator
    let strm = strm
        .then(|item| async move {
            match item {
                Ok(progress) => {
                    let comment = progress.comment.to_string();
                    let current = progress.current_task;
                    let total = progress.total_task;
                    let result = if current == total {
                        let result_inner = progress.result.expect("last event must carry result");
                        let obj: Py<PyAny> = Python::with_gil(|py| {
                            Py::new(py, T::from_inner(result_inner)).map(|v| v.into_any())
                        })
                        .map_err(|e| e.to_string())?;
                        Some(obj)
                    } else {
                        None
                    };
                    Ok(PyCacheProgress {
                        comment,
                        current,
                        total,
                        result,
                    })
                }
                Err(e) => Err(e),
            }
        })
        .boxed();

    Ok(PyCacheProgressSyncIterator { rt, strm })
}

#[pyclass(unsendable, name = "CacheProgressIterator")]
pub struct PyCacheProgressIterator {
    rx: async_channel::Receiver<Result<PyCacheProgress, String>>,
}

#[pymethods]
impl PyCacheProgressIterator {
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

pub fn create_cache_progress_iterator<T>(cache_key: impl Into<String>) -> PyCacheProgressIterator
where
    T: PyWrapper,
    T::Inner: TryFromCache,
{
    let rt = pyo3_async_runtimes::tokio::get_runtime();
    let (tx, rx) = async_channel::unbounded::<Result<PyCacheProgress, String>>();
    let cache_key = cache_key.into();

    rt.spawn(async move {
        let mut strm: BoxStream<'static, Result<CacheProgress<T::Inner>, String>> =
            Box::pin(Cache::new().try_create::<T::Inner>(cache_key));

        while let Some(item) = strm.next().await {
            let out = match item {
                Ok(mut progress) => {
                    let comment = progress.comment.to_string();
                    let current = progress.current_task;
                    let total = progress.total_task;
                    let result = if current == total {
                        match progress.result.take() {
                            Some(inner) => {
                                // T::Inner -> Py<PyAny>
                                match Python::with_gil(|py| {
                                    Py::new(py, T::from_inner(inner)).map(|o| o.into_any())
                                }) {
                                    Ok(obj) => Some(obj),
                                    Err(e) => {
                                        let _ = tx.send(Err(e.to_string())).await;
                                        continue;
                                    }
                                }
                            }
                            None => None,
                        }
                    } else {
                        None
                    };

                    Ok(PyCacheProgress {
                        comment,
                        current,
                        total,
                        result,
                    })
                }
                Err(e) => Err(e),
            };

            if tx.send(out).await.is_err() {
                break; // Exit if consumer vanished
            }
        }
    });

    PyCacheProgressIterator { rx }
}
