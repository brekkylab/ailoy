use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::pydict_to_json,
    tool::{BuiltinTool, Tool, create_terminal_tool},
    value::{ToolCallArg, ToolDesc},
};

#[gen_stub_pyclass]
#[pyclass(name = "BuiltinTool")]
pub struct PyBuiltinTool {
    inner: BuiltinTool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBuiltinTool {
    #[staticmethod]
    pub fn terminal() -> PyResult<Self> {
        Ok(PyBuiltinTool {
            inner: create_terminal_tool(),
        })
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        Ok(self.inner.get_description())
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn run(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let args = if let Some(kwargs) = kwargs {
            serde_json::from_value::<ToolCallArg>(pydict_to_json(py, kwargs).unwrap())
                .expect("args is not a valid ToolCallArg")
        } else {
            ToolCallArg::new_null()
        };

        let inner = self.inner.clone();
        let fut = async move {
            let results = inner.run(args).await;
            match results {
                Ok(parts) => Ok(parts),
                Err(e) => Err(PyRuntimeError::new_err(e)),
            }
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}
