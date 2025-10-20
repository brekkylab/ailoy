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
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
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
