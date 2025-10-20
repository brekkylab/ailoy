mod builtin;
mod function;
mod mcp;

use std::{fmt::Debug, sync::Arc};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
pub use builtin::*;
pub use function::*;
pub use mcp::*;

use crate::{
    knowledge::KnowledgeTool,
    value::{ToolDesc, Value},
};

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait ToolBehavior: Debug + Clone {
    fn get_description(&self) -> ToolDesc;
    async fn run(&self, args: Value) -> anyhow::Result<Value>;
}

#[derive(Debug, Clone)]
pub enum ToolInner {
    Function(FunctionTool),
    MCP(MCPTool),
    Knowledge(KnowledgeTool),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(subclass))]
pub struct Tool {
    inner: ToolInner,
}

impl Tool {
    pub fn new_function(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self {
            inner: ToolInner::Function(FunctionTool::new(desc, f)),
        }
    }

    pub fn new_mcp(tool: MCPTool) -> Self {
        Self {
            inner: ToolInner::MCP(tool),
        }
    }

    pub fn new_knowledge(tool: KnowledgeTool) -> Self {
        Self {
            inner: ToolInner::Knowledge(tool),
        }
    }
}

#[multi_platform_async_trait]
impl ToolBehavior for Tool {
    fn get_description(&self) -> ToolDesc {
        match &self.inner {
            ToolInner::Function(tool) => tool.get_description(),
            ToolInner::MCP(tool) => tool.get_description(),
            ToolInner::Knowledge(tool) => tool.get_description(),
        }
    }

    async fn run(&self, args: Value) -> anyhow::Result<Value> {
        match &self.inner {
            ToolInner::Function(tool) => tool.run(args).await,
            ToolInner::MCP(tool) => tool.run(args).await,
            ToolInner::Knowledge(tool) => tool.run(args).await,
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use std::sync::Arc;

    use indexmap::IndexMap;
    use ordered_float::OrderedFloat;
    use pyo3::{
        Bound, IntoPyObjectExt, Py, PyAny, PyResult, Python,
        prelude::*,
        pymethods,
        types::{PyAnyMethods, PyBool, PyDict, PyFloat, PyList, PyListMethods, PyString, PyType},
    };
    use pyo3_stub_gen_derive::gen_stub_pymethods;

    use super::*;
    use crate::value::Value;

    fn value_to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        match value {
            Value::Null => py.None().into_bound_py_any(py),
            Value::Bool(b) => PyBool::new(py, *b).into_bound_py_any(py),
            Value::Unsigned(u) => u.into_bound_py_any(py),
            Value::Integer(i) => i.into_bound_py_any(py),
            Value::Float(f) => PyFloat::new(py, f.0).into_bound_py_any(py),
            Value::String(s) => PyString::new(py, s).into_bound_py_any(py),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    py_list.append(value_to_python(py, item)?)?;
                }
                py_list.into_bound_py_any(py)
            }
            Value::Object(map) => {
                let py_dict = PyDict::new(py);
                for (key, val) in map {
                    py_dict.set_item(key, value_to_python(py, val)?)?;
                }
                py_dict.into_bound_py_any(py)
            }
        }
    }

    fn python_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
        if obj.is_none() {
            Ok(Value::Null)
        } else if let Ok(b) = obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = obj.extract::<i64>() {
            Ok(Value::Integer(i))
        } else if let Ok(u) = obj.extract::<u64>() {
            Ok(Value::Unsigned(u))
        } else if let Ok(f) = obj.extract::<f64>() {
            Ok(Value::Float(OrderedFloat(f)))
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(list) = obj.downcast::<PyList>() {
            let mut arr = Vec::new();
            for item in list.iter() {
                arr.push(python_to_value(&item)?);
            }
            Ok(Value::Array(arr))
        } else if let Ok(dict) = obj.downcast::<PyDict>() {
            let mut map = IndexMap::new();
            for (key, val) in dict.iter() {
                let key_str = key.extract::<String>()?;
                map.insert(key_str, python_to_value(&val)?);
            }
            Ok(Value::Object(map))
        } else {
            Ok(Value::String(obj.to_string()))
        }
    }

    pub fn wrap_python_function(py_func: Py<PyAny>) -> Arc<dyn Fn(Value) -> Value + Send + Sync> {
        Arc::new(move |args: Value| -> Value {
            Python::attach(|py| {
                let py_func_bound = py_func.clone_ref(py);

                // Rust Value -> Python object
                let py_args = match value_to_python(py, &args) {
                    Ok(obj) => obj,
                    Err(e) => {
                        return Value::object([(
                            "error",
                            format!("Failed to convert arguments to Python: {}", e),
                        )]);
                    }
                };

                // call python function
                let result = if let Ok(dict) = py_args.downcast::<PyDict>() {
                    py_func_bound.bind(py).call((), Some(&dict))
                } else {
                    return Value::object([(
                        "error",
                        format!("Failed to convert arguments to Python dict: {:?}", py_args),
                    )]);
                };

                let py_result = match result {
                    Ok(r) => r,
                    Err(e) => {
                        return Value::object([(
                            "error",
                            format!("Python function call failed: {}", e),
                        )]);
                    }
                };

                // check whether result is coroutine
                let is_coroutine = py_result.hasattr("__await__").unwrap_or(false);

                let final_result = if is_coroutine {
                    // execute coroutine using Python `asyncio`
                    match py
                        .import("asyncio")
                        .and_then(|asyncio| asyncio.call_method1("run", (py_result,)))
                    {
                        Ok(r) => r.unbind(),
                        Err(e) => {
                            return Value::object([(
                                "error",
                                format!("Async execution failed: {}", e),
                            )]);
                        }
                    }
                } else {
                    py_result.unbind()
                };

                // Python object -> Rust Value
                match python_to_value(&final_result.bind(py)) {
                    Ok(value) => value,
                    Err(e) => Value::object([(
                        "error",
                        format!("Failed to convert Python result to Value: {}", e),
                    )]),
                }
            })
        })
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl Tool {
        #[classmethod]
        #[pyo3(signature = (desc, func))]
        pub fn new_py_function(_cls: &Bound<'_, PyType>, desc: ToolDesc, func: Py<PyAny>) -> Self {
            Self {
                inner: ToolInner::Function(FunctionTool::new(desc, wrap_python_function(func))),
            }
        }

        pub fn __repr__(&self) -> String {
            match &self.inner {
                ToolInner::Function(tool) => {
                    format!("Tool(FunctionTool(name={}))", tool.get_description().name)
                }
                ToolInner::MCP(tool) => {
                    format!("Tool(MCPTool(name={}))", tool.get_description().name)
                }
                ToolInner::Knowledge(tool) => {
                    format!("Tool(KnowledgeTool(name={}))", tool.get_description().name)
                }
            }
        }

        #[pyo3(name = "get_description", signature=())]
        fn get_description_py(&self) -> ToolDesc {
            self.get_description()
        }

        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[typing.Any]"))]
        #[pyo3(signature = (**kwargs))]
        fn __call__<'py>(
            &self,
            py: Python<'py>,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            self.call(py, kwargs)
        }

        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[typing.Any]"))]
        #[pyo3(signature = (**kwargs))]
        fn call<'py>(
            &self,
            py: Python<'py>,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            // Python object -> Rust Value
            let py_input = kwargs
                .map(|py_dict| py_dict.clone().into_any())
                .unwrap_or(PyDict::new(py).into_any());
            let input = python_to_value(&py_input).unwrap_or_else(|e| {
                Value::object([(
                    "error",
                    format!("Failed to convert Python result to Value: {}", e),
                )])
            });

            // create Rust Future to run tool
            let inner = self.inner.clone();
            let future = async move {
                let result = match inner {
                    ToolInner::Function(tool) => tool.run(input).await,
                    ToolInner::MCP(tool) => tool.run(input).await,
                    ToolInner::Knowledge(tool) => tool.run(input).await,
                }
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                // Rust Value -> Python object
                Python::attach(|py| value_to_python(py, &result).map(|bound| bound.unbind()))
            };

            // Rust Future -> Python Coroutine
            pyo3_async_runtimes::tokio::future_into_py(py, future)
        }

        #[pyo3(signature = (**kwargs))]
        fn call_sync(
            &self,
            py: Python<'_>,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Py<PyAny>> {
            // Python object -> Rust Value
            let py_input = kwargs
                .map(|py_dict| py_dict.clone().into_any())
                .unwrap_or(PyDict::new(py).into_any());
            let input = python_to_value(&py_input).unwrap_or_else(|e| {
                Value::object([(
                    "error",
                    format!("Failed to convert Python result to Value: {}", e),
                )])
            });

            // run tool synced
            let result = match tokio::runtime::Handle::try_current() {
                Ok(handle) => tokio::task::block_in_place(|| handle.block_on(self.run(input))),
                Err(_) => tokio::runtime::Runtime::new()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                    .block_on(self.run(input)),
            }
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Rust Value -> Python object
            value_to_python(py, &result).map(|bound| bound.unbind())
        }

        #[staticmethod]
        fn terminal() -> Self {
            create_terminal_tool()
        }
    }
}
