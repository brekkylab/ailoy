use std::collections::HashSet;

use futures::StreamExt;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_stub_gen::{PyStubType, TypeInfo, derive::*};

use crate::cache::{Cache, TryFromCache};

#[pyclass(subclass)]
pub struct GenericCacheResultT {}

impl PyStubType for GenericCacheResultT {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: format!("typing.Generic[CacheResultT]"),
            import: HashSet::new(),
        }
    }
}

pub struct CacheResultT(pub Py<PyAny>);

impl<'py> IntoPyObject<'py> for &CacheResultT {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.bind(py).clone())
    }
}

impl PyStubType for CacheResultT {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: format!("CacheResultT"),
            import: HashSet::new(),
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "CacheProgress", extends = GenericCacheResultT)]
pub struct PyCacheProgress {
    #[pyo3(get)]
    pub comment: String,

    #[pyo3(get)]
    pub current: usize,

    #[pyo3(get)]
    pub total: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCacheProgress {
    fn __repr__(&self) -> String {
        format!(
            "CacheProgress(comment=\"{}\", current={}, total={})",
            self.comment, self.current, self.total
        )
    }
}

pub async fn await_cache_result<T>(
    cache_key: impl Into<String>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<T>
where
    T: TryFromCache + 'static,
{
    let cache_key = cache_key.into();
    let mut strm = Box::pin(Cache::new().try_create::<T>(cache_key));
    while let Some(item) = strm.next().await {
        if item.is_err() {
            // Exit the loop and return the error
            return item.err().map(|e| Err(PyRuntimeError::new_err(e))).unwrap();
        }

        let progress = item.unwrap();

        // Call progress_callback if exists
        if let Some(callback) = &progress_callback {
            Python::attach(|py| {
                let py_obj = Py::new(
                    py,
                    (
                        PyCacheProgress {
                            comment: progress.comment.clone(),
                            current: progress.current_task,
                            total: progress.total_task,
                        },
                        GenericCacheResultT {},
                    ),
                )?;
                callback.call1(py, (py_obj,))
            })?;
        }

        if progress.current_task < progress.total_task {
            // Continue if progress is not completed
            continue;
        }

        match progress.result {
            Some(inner) => return Ok(inner),
            None => {
                return Err(PyRuntimeError::new_err(
                    "CacheProgress didn't return anything",
                ));
            }
        }
    }

    Err(PyRuntimeError::new_err("Unreachable"))
}
