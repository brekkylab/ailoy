use std::sync::Arc;

use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::{PyWrapper, json_to_pydict, pydict_to_json},
    tool::{
        BuiltinTool, MCPTransport, Tool, create_terminal_tool,
        mcp::native::{MCPTool, mcp_tools_from_stdio, mcp_tools_from_streamable_http},
    },
    value::{Part, ToolCallArg, ToolDesc},
};

#[gen_stub_pyclass]
#[pyclass(name = "Tool", subclass)]
pub struct PyTool {}

#[gen_stub_pymethods]
#[pymethods]
impl PyTool {
    #[gen_stub(skip)]
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        Err(PyNotImplementedError::new_err(
            "Tool subclasses must implement 'description'",
        ))
    }

    #[allow(unused_variables)]
    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn run(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        Err(PyNotImplementedError::new_err(
            "Tool subclasses must implement 'run'",
        ))
    }
}

pub trait PyToolMethods<T: Tool + Clone + 'static> {
    fn inner(&self) -> &T;

    fn get_description(&self) -> PyResult<ToolDesc> {
        Ok(self.inner().get_description())
    }

    fn call(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let args = if let Some(kwargs) = kwargs {
            serde_json::from_value::<ToolCallArg>(pydict_to_json(py, kwargs).unwrap())
                .expect("args is not a valid ToolCallArg")
        } else {
            ToolCallArg::new_null()
        };

        let inner = self.inner().clone();
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

#[gen_stub_pyclass]
#[pyclass(name = "BuiltinTool", extends = PyTool)]
pub struct PyBuiltinTool {
    inner: BuiltinTool,
}

impl PyWrapper for PyBuiltinTool {
    type Inner = BuiltinTool;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyTool {};
        let child = Self { inner };
        Py::new(py, (child, base))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyToolMethods<BuiltinTool> for PyBuiltinTool {
    fn inner(&self) -> &BuiltinTool {
        &self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBuiltinTool {
    #[staticmethod]
    fn terminal(py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyTool {};
        let child = Self {
            inner: create_terminal_tool(),
        };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        self.get_description()
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        self.call(py, kwargs)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "MCPTool", extends = PyTool)]
pub struct PyMCPTool {
    inner: MCPTool,
}

impl PyWrapper for PyMCPTool {
    type Inner = MCPTool;

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyTool {};
        let child = Self { inner };
        Py::new(py, (child, base))
    }

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }
}

impl PyToolMethods<MCPTool> for PyMCPTool {
    fn inner(&self) -> &MCPTool {
        &self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMCPTool {
    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        self.get_description()
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        self.call(py, kwargs)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MCPTransport {
    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[MCPTool]]"))]
    fn tools(&self, py: Python<'_>, tool_name_prefix: String) -> PyResult<Py<PyAny>> {
        let self_clone = self.clone();
        let fut = async move {
            match self_clone {
                MCPTransport::Stdio { command, args } => {
                    mcp_tools_from_stdio(command.clone(), args.clone(), tool_name_prefix.as_str())
                        .await
                }
                MCPTransport::StreamableHttp { url } => {
                    mcp_tools_from_streamable_http(url.as_str(), tool_name_prefix.as_str()).await
                }
            }
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_iter()
            .map(|t| {
                Python::attach(|py| {
                    Ok(Py::new(py, (PyMCPTool { inner: t }, PyTool::new()))?
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind())
                })
            })
            .collect::<PyResult<Vec<Py<PyAny>>>>()
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}

#[derive(Debug, Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "PythonFunctionTool", extends = PyTool)]
pub struct PythonFunctionTool {
    desc: ToolDesc,
    // Sync tool function
    func: Arc<Py<PyAny>>,
}

#[async_trait::async_trait]
impl Tool for PythonFunctionTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: ToolCallArg) -> Result<Vec<Part>, String> {
        let func = &self.func;
        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let kwargs = json_to_pydict(py, &serde_json::to_value(args).unwrap())?;
            let result = func.call(py, PyTuple::empty(py), Some(&kwargs))?;
            Ok(result)
        })
        .map_err(|e| e.to_string())?;

        let parts = Python::attach(|py| {
            if let Ok(result) = result.downcast_bound::<PyList>(py) {
                result
                    .iter()
                    .map(|item| {
                        if let Ok(item) = item.downcast::<Part>() {
                            Ok(item.as_unbound().borrow(py).clone())
                        } else {
                            Err(PyRuntimeError::new_err("Tool item is not a type of Part"))
                        }
                    })
                    .collect::<PyResult<Vec<Part>>>()
            } else {
                Err(PyRuntimeError::new_err(
                    "Tool result should be a list of Part",
                ))
            }
        })
        .map_err(|e| e.to_string())?;

        Ok(parts)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PythonFunctionTool {
    #[new]
    fn __new__(
        py: Python<'_>,
        description: ToolDesc,
        #[gen_stub(override_type(type_repr = "typing.Callable[..., list[Part]]"))] func: Py<PyAny>,
    ) -> PyResult<Py<Self>> {
        let base = PyTool {};
        let child = PythonFunctionTool {
            desc: description,
            func: Arc::new(func),
        };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        Ok(self.get_description())
    }

    #[gen_stub(override_return_type(type_repr = "list[Part]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let result = self.func.call(py, PyTuple::empty(py), kwargs)?;
        Ok(result)
    }
}

#[derive(Debug, Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "PythonAsyncFunctionTool", extends = PyTool)]
pub struct PythonAsyncFunctionTool {
    desc: ToolDesc,
    // Async tool function
    func: Arc<Py<PyAny>>,
}

#[async_trait::async_trait]
impl Tool for PythonAsyncFunctionTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: ToolCallArg) -> Result<Vec<Part>, String> {
        let func = &self.func;
        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let asyncio = py.import("asyncio")?;

            let kwargs = json_to_pydict(py, &serde_json::to_value(args).unwrap()).unwrap();
            let coro = func.call(py, PyTuple::empty(py), Some(&kwargs))?;

            let result = if let Ok(loop_obj) = asyncio.call_method0("get_running_loop") {
                let task = asyncio.call_method1("create_task", (coro,))?;
                loop_obj.call_method1("run_until_complete", (task,))?
            } else {
                asyncio.call_method1("run", (coro,))?
            };

            Ok(result.into())
        })
        .map_err(|e| e.to_string())?;

        let parts = Python::attach(|py| {
            if let Ok(result) = result.downcast_bound::<PyList>(py) {
                result
                    .iter()
                    .map(|item| {
                        if let Ok(item) = item.downcast::<Part>() {
                            Ok(item.as_unbound().borrow(py).clone())
                        } else {
                            Err(PyRuntimeError::new_err("Tool item is not a type of Part"))
                        }
                    })
                    .collect::<PyResult<Vec<Part>>>()
            } else {
                Err(PyRuntimeError::new_err(
                    "Tool result should be a list of Part",
                ))
            }
        })
        .map_err(|e| e.to_string())?;

        Ok(parts)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PythonAsyncFunctionTool {
    #[new]
    fn __new__(
        py: Python<'_>,
        description: ToolDesc,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[..., typing.Awaitable[list[Part]]]"
        ))]
        func: Py<PyAny>,
    ) -> PyResult<Py<Self>> {
        let base = PyTool {};
        let child = PythonAsyncFunctionTool {
            desc: description,
            func: Arc::new(func),
        };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        Ok(self.get_description())
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let args = if let Some(kwargs) = kwargs {
            serde_json::from_value::<ToolCallArg>(pydict_to_json(py, kwargs).unwrap())
                .expect("args is not a valid ToolCallArg")
        } else {
            ToolCallArg::new_null()
        };

        let self_clone = self.clone();
        let fut = async move {
            let parts = self_clone
                .run(args)
                .await
                .map_err(|e| PyRuntimeError::new_err(e))?;
            let parts_any = parts
                .into_iter()
                .map(|p| Python::attach(|py| Ok(p.into_pyobject(py).unwrap().into_any().unbind())))
                .collect::<PyResult<Vec<Py<PyAny>>>>();
            parts_any
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}
