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
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", subclass))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Tool {
    inner: ToolInner,
}

impl Tool {
    pub fn new_builtin(kind: BuiltinToolKind, config: Value) -> anyhow::Result<Self> {
        let tool = match kind {
            BuiltinToolKind::Terminal => create_terminal_tool(config),
            BuiltinToolKind::WebSearchDuckduckgo => create_web_search_duckduckgo_tool(config),
            BuiltinToolKind::WebFetch => create_web_fetch_tool(config),
        }?;
        Ok(Self {
            inner: ToolInner::Function(tool),
        })
    }

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

    use pyo3::{
        Bound, Py, PyAny, PyResult, Python,
        exceptions::PyRuntimeError,
        prelude::*,
        pymethods,
        types::{PyAnyMethods, PyDict, PyType},
    };
    use pyo3_stub_gen_derive::gen_stub_pymethods;

    use super::*;
    use crate::{
        ffi::py::base::{python_to_value, value_to_python},
        value::Value,
    };

    pub fn wrap_python_function(py_func: Py<PyAny>) -> Arc<ToolFunc> {
        let f: Box<ToolFunc> = Box::new(move |args: Value| {
            let py_func = Python::attach(|py| py_func.clone_ref(py));
            Box::pin(async move {
                let (py_result, is_awaitable) = Python::attach(|py| {
                    // Rust Value -> Python dict
                    let py_args = value_to_python(py, &args).unwrap();
                    let kwargs = py_args.cast::<PyDict>().unwrap();

                    // call python function
                    let result = py_func.bind(py).call((), Some(kwargs)).unwrap();

                    // check whether result is coroutine
                    let is_awaitable = result.hasattr("__await__").unwrap_or(false);
                    (result.unbind(), is_awaitable)
                });

                // await function result if awaitable
                let final_result = if is_awaitable {
                    let fut = Python::attach(|py| {
                        pyo3_async_runtimes::tokio::into_future(py_result.bind(py).to_owned())
                    })?;
                    fut.await?
                } else {
                    py_result
                };

                // Python object -> Rust Value
                Python::attach(|py| python_to_value(&final_result.bind(py)).map_err(Into::into))
            })
        });
        Arc::new(f)
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl Tool {
        #[classmethod]
        #[pyo3(name = "new_builtin", signature = (kind, **kwargs))]
        pub fn new_builtin_py(
            _cls: &Bound<'_, PyType>,
            kind: BuiltinToolKind,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let tool_config = if let Some(kwargs) = kwargs {
                python_to_value(kwargs).unwrap()
            } else {
                Value::object_empty()
            };
            Self::new_builtin(kind, tool_config).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        #[gen_stub(skip)]
        #[classmethod]
        #[pyo3(name = "__new_py_function__", signature = (desc, func))]
        pub fn __new_py_function__(
            _cls: &Bound<'_, PyType>,
            desc: ToolDesc,
            func: Py<PyAny>,
        ) -> Self {
            Self {
                inner: ToolInner::Function(FunctionTool::new(desc, wrap_python_function(func))),
            }
        }

        #[allow(unused_variables)]
        #[classmethod]
        #[pyo3(name = "new_py_function", signature = (func, desc = None))]
        pub fn new_py_function(
            _cls: &Bound<'_, PyType>,
            func: Py<PyAny>,
            desc: Option<ToolDesc>,
        ) -> Self {
            unimplemented!("This classmethod will be monkeypatched in Python")
        }

        fn __repr__(&self) -> String {
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
            let tool = self.clone();
            let future = async move {
                let result = tool.run(input).await?;
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
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::{Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction};
    use napi_derive::napi;
    use rmcp::transport::ConfigureCommandExt;

    use super::*;

    #[napi]
    impl Tool {
        #[napi(js_name = "newBuiltin")]
        pub fn new_builtin_js(kind: BuiltinToolKind, config: Option<Value>) -> napi::Result<Self> {
            Self::new_builtin(kind, config.unwrap_or(Value::object_empty()))
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "newFunction")]
        pub fn new_function_js(
            desc: ToolDesc,
            func: ThreadsafeFunction<Value, Promise<Value>, Value, Status, false, false>,
        ) -> Self {
            let func = Arc::new(func);
            let tool_func: Box<ToolFunc> = Box::new(move |value: Value| {
                let func = func.clone();
                Box::pin(async move {
                    let promise = func.call_async(value).await?;
                    match promise.await {
                        Ok(result) => Ok(result),
                        Err(e) => Err(anyhow::anyhow!(e.to_string())),
                    }
                })
            });
            Self::new_function(desc, Arc::new(tool_func))
        }

        #[napi(getter, js_name = "description")]
        pub fn description_js(&self) -> ToolDesc {
            self.get_description().clone()
        }

        #[napi(js_name = "run")]
        pub async fn run_js(&self, args: Value) -> napi::Result<Value> {
            self.run(args)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }
    }

    #[napi]
    impl MCPClient {
        #[napi(js_name = "newStdio")]
        pub async fn new_stdio_js(command: String, args: Vec<String>) -> napi::Result<Self> {
            let command = tokio::process::Command::new(command).configure(|cmd| {
                cmd.args(args);
            });
            MCPClient::from_stdio(command)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "newStreamableHttp")]
        pub async fn new_streamable_http_js(url: String) -> napi::Result<Self> {
            Self::from_streamable_http(url)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(getter, js_name = "tools")]
        pub fn tools_js(&self) -> Vec<Tool> {
            self.get_tools()
                .into_iter()
                .map(|t| Tool::new_mcp(t.clone()))
                .collect()
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use std::sync::Arc;

    use anyhow::anyhow;
    use js_sys::Promise;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    use super::*;

    #[wasm_bindgen]
    impl Tool {
        #[wasm_bindgen(js_name = "newBuiltin")]
        pub fn new_builtin_js(
            kind: BuiltinToolKind,
            config: Option<Value>,
        ) -> Result<Self, js_sys::Error> {
            Self::new_builtin(kind, config.unwrap_or(Value::object_empty()))
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "newFunction")]
        pub fn new_function_js(
            desc: ToolDesc,
            #[wasm_bindgen(unchecked_param_type = "(args: any) => Promise<any>")]
            func: js_sys::Function,
        ) -> Self {
            let tool_func: Box<ToolFunc> = Box::new(move |value: Value| {
                let func = func.clone();
                Box::pin(async move {
                    let js_val: JsValue = value.into();
                    let js_promise = func.call1(&JsValue::NULL, &js_val).map_err(|e| {
                        anyhow!(
                            e.as_string()
                                .unwrap_or("Failed to call tool function".to_owned())
                        )
                    })?;
                    let js_future = JsFuture::from(Promise::from(js_promise));
                    let js_ret: JsValue = js_future.await.map_err(|e| {
                        anyhow!(e.as_string().unwrap_or("Failed to await future".to_owned()))
                    })?;
                    js_ret.try_into().map_err(|e: js_sys::Error| {
                        anyhow!(e.as_string().unwrap_or(
                            "Failed to convert tool function result to js value".to_owned()
                        ))
                    })
                })
            });
            Self::new_function(desc, Arc::new(tool_func))
        }

        #[wasm_bindgen(getter)]
        pub fn description(&self) -> ToolDesc {
            self.get_description()
        }

        #[wasm_bindgen(js_name = "run")]
        pub async fn run_js(&self, args: Value) -> Result<Value, js_sys::Error> {
            self.run(args)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }
    }

    #[wasm_bindgen]
    impl MCPClient {
        #[wasm_bindgen(js_name = "streamableHttp")]
        pub async fn from_streamable_http_js(url: String) -> Result<Self, js_sys::Error> {
            Self::from_streamable_http(url)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(getter, js_name = "tools")]
        pub fn tools_js(&self) -> Vec<Tool> {
            self.tools
                .clone()
                .into_iter()
                .map(|t| Tool::new_mcp(t))
                .collect()
        }
    }
}
