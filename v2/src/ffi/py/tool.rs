use std::sync::Arc;

use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError},
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

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "Tool", subclass)]
pub struct PyTool {}

#[gen_stub_pymethods]
#[pymethods]
impl PyTool {
    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        Err(PyNotImplementedError::new_err(
            "Tool subclasses must implement 'description' getter",
        ))
    }

    /// This is not a function used by Agents, but this let users directly call the tool function for debugging purpose.
    #[allow(unused_variables)]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        Err(PyNotImplementedError::new_err(
            "Tool subclasses must implement '__call__'",
        ))
    }
}

pub trait PyToolMethods<T: Tool + Clone + 'static> {
    fn inner(&self) -> &T;

    fn _description(&self) -> PyResult<ToolDesc> {
        Ok(self.inner().get_description())
    }

    fn call(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let args = if let Some(kwargs) = kwargs {
            serde_json::from_value::<ToolCallArg>(serde_json::Value::Object(pydict_to_json(
                py, kwargs,
            )?))
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
        Py::new(py, (Self { inner }, PyTool {}))
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
        Py::new(
            py,
            (
                Self {
                    inner: create_terminal_tool(),
                },
                PyTool {},
            ),
        )
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        self._description()
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        self.call(py, kwargs)
    }

    fn __repr__(&self) -> String {
        format!(
            "BuiltinTool(name=\"{}\")",
            self.inner.get_description().name
        )
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
        Py::new(py, (Self { inner }, PyTool {}))
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
        self._description()
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        self.call(py, kwargs)
    }

    fn __repr__(&self) -> String {
        format!("MCPTool(name=\"{}\")", self.inner.get_description().name)
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
                    Ok(Py::new(py, (PyMCPTool { inner: t }, PyTool {}))?
                        .into_pyobject(py)?
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

fn pylist_to_parts(list: Py<PyAny>) -> Result<Vec<Part>, String> {
    let parts = Python::attach(|py| {
        if let Ok(list) = list.downcast_bound::<PyList>(py) {
            list.iter()
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
        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let kwargs = match json_to_pydict(py, &serde_json::to_value(args).unwrap())? {
                Some(kwargs) => Some(&kwargs.clone()),
                None => None,
            };
            let result = self.func.call(py, PyTuple::empty(py), kwargs)?;
            Ok(result)
        })
        .map_err(|e| e.to_string())?;

        pylist_to_parts(result)
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
        Py::new(
            py,
            (
                PythonFunctionTool {
                    desc: description,
                    func: Arc::new(func),
                },
                PyTool {},
            ),
        )
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        Ok(self.get_description())
    }

    /// Unlike another tools, this tool's __call__ is executed synchronously.
    #[gen_stub(override_return_type(type_repr = "list[Part]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        self.func.call(py, PyTuple::empty(py), kwargs)
    }

    fn __repr__(&self) -> String {
        format!(
            "PythonFunctionTool(name=\"{}\")",
            self.get_description().name
        )
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

            let kwargs = match json_to_pydict(py, &serde_json::to_value(args).unwrap())? {
                Some(kwargs) => Some(&kwargs.clone()),
                None => None,
            };
            let coro = func.call(py, PyTuple::empty(py), kwargs)?;

            let result = if let Ok(loop_obj) = asyncio.call_method0("get_running_loop") {
                let task = asyncio.call_method1("create_task", (coro,))?;
                loop_obj.call_method1("run_until_complete", (task,))?
            } else {
                asyncio.call_method1("run", (coro,))?
            };

            Ok(result.into())
        })
        .map_err(|e| e.to_string())?;

        pylist_to_parts(result)
    }
}

impl PyToolMethods<PythonAsyncFunctionTool> for PythonAsyncFunctionTool {
    fn inner(&self) -> &PythonAsyncFunctionTool {
        self
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
        Py::new(
            py,
            (
                PythonAsyncFunctionTool {
                    desc: description,
                    func: Arc::new(func),
                },
                PyTool {},
            ),
        )
    }

    #[getter]
    fn description(&self) -> PyResult<ToolDesc> {
        self._description()
    }

    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[list[Part]]"))]
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        self.call(py, kwargs)
    }

    fn __repr__(&self) -> String {
        format!(
            "PythonAsyncFunctionTool(name=\"{}\")",
            self.get_description().name
        )
    }
}

pub struct ArcTool(pub Arc<dyn Tool>);

impl<'py> IntoPyObject<'py> for ArcTool {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_any = if let Some(tool) = self.0.downcast_ref::<BuiltinTool>() {
            Ok(PyBuiltinTool::into_py_obj(tool.clone(), py)?.into_any())
        } else if let Some(tool) = self.0.downcast_ref::<MCPTool>() {
            Ok(PyMCPTool::into_py_obj(tool.clone(), py)?.into_any())
        } else if let Some(tool) = self.0.downcast_ref::<PythonFunctionTool>() {
            Ok(Py::new(py, (tool.clone(), PyTool {}))?.into_any())
        } else if let Some(tool) = self.0.downcast_ref::<PythonAsyncFunctionTool>() {
            Ok(Py::new(py, (tool.clone(), PyTool {}))?.into_any())
        } else {
            Err(PyRuntimeError::new_err("Failed to downcast Tool"))
        }?;
        Ok(py_any.into_bound(py))
    }
}

impl<'py> FromPyObject<'py> for ArcTool {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            if let Ok(tool) = ob.downcast::<PyBuiltinTool>() {
                Ok(ArcTool(
                    Arc::new(tool.as_unbound().borrow(py).inner().clone()) as Arc<dyn Tool>,
                ))
            } else if let Ok(tool) = ob.downcast::<PyMCPTool>() {
                Ok(ArcTool(
                    Arc::new(tool.as_unbound().borrow(py).inner().clone()) as Arc<dyn Tool>,
                ))
            } else if let Ok(tool) = ob.downcast::<PythonFunctionTool>() {
                Ok(ArcTool(
                    Arc::new(tool.as_unbound().borrow(py).clone()) as Arc<dyn Tool>
                ))
            } else if let Ok(tool) = ob.downcast::<PythonAsyncFunctionTool>() {
                Ok(ArcTool(
                    Arc::new(tool.as_unbound().borrow(py).clone()) as Arc<dyn Tool>
                ))
            } else {
                Err(PyTypeError::new_err("Unknown tool object provided"))
            }
        })
    }
}
