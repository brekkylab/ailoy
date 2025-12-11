use std::collections::HashSet;

use futures::StreamExt;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_stub_gen::{PyStubType, TypeInfo, derive::*};

use crate::utils::BoxStream;

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
#[pyclass(module = "ailoy._core", name = "CacheProgress")]
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
    pub fn __repr__(&self) -> String {
        format!(
            "CacheProgress(comment=\"{}\", current={}, total={})",
            self.comment, self.current, self.total
        )
    }
}

pub async fn await_cache_result<'a, T>(
    mut cache_strm: BoxStream<'a, anyhow::Result<crate::cache::CacheProgress<T>>>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<T> {
    while let Some(item) = cache_strm.next().await {
        let progress = item?;

        // Call progress_callback if exists
        if let Some(callback) = &progress_callback {
            Python::attach(|py| {
                let py_obj = Py::new(
                    py,
                    PyCacheProgress {
                        comment: progress.comment.clone(),
                        current: progress.current_task,
                        total: progress.total_task,
                    },
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
