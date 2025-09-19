use std::sync::Arc;

use futures::lock::Mutex;
use napi::{Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction};
use napi_derive::napi;
use serde_json::{Map, Value};

use crate::{
    ffi::node::{common::await_future, value::JsPart},
    tool::{
        BuiltinTool, MCPTransport, Tool, create_terminal_tool,
        mcp::native::MCPTool,
        native::{mcp_tools_from_stdio, mcp_tools_from_streamable_http},
    },
    value::{ToolCallArg, ToolDesc},
};

pub trait JsToolMethods<T: Tool + 'static> {
    fn inner(&self) -> Arc<Mutex<T>>;

    fn _description(&self) -> napi::Result<ToolDesc> {
        await_future(async { Ok(self.inner().lock().await.get_description()) })
    }

    async fn _call(&self, kwargs: Option<Map<String, Value>>) -> napi::Result<Vec<JsPart>> {
        let args = if let Some(kwargs) = kwargs {
            let value = serde_json::Value::Object(kwargs);
            serde_json::from_value::<ToolCallArg>(value).expect("args is not a valid ToolCallArg")
        } else {
            ToolCallArg::new_null()
        };

        let inner = self.inner();
        let results = inner.lock().await.run(args).await;
        match results {
            Ok(parts) => Ok(parts.into_iter().map(|part| part.into()).collect()),
            Err(e) => Err(napi::Error::new(Status::GenericFailure, e)),
        }
    }
}

#[napi(js_name = "BuiltinTool")]
pub struct JsBuiltinTool {
    inner: Arc<Mutex<BuiltinTool>>,
}

impl JsToolMethods<BuiltinTool> for JsBuiltinTool {
    fn inner(&self) -> Arc<Mutex<BuiltinTool>> {
        self.inner.clone()
    }
}

#[napi]
impl JsBuiltinTool {
    #[napi(factory)]
    pub fn terminal() -> napi::Result<Self> {
        let inner = create_terminal_tool();
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[napi(getter)]
    pub fn description(&self) -> napi::Result<ToolDesc> {
        self._description()
    }

    #[napi]
    pub async fn call(&self, kwargs: Option<Map<String, Value>>) -> napi::Result<Vec<JsPart>> {
        self._call(kwargs).await
    }
}

#[napi(js_name = "MCPTool")]
pub struct JsMCPTool {
    inner: Arc<Mutex<MCPTool>>,
}

impl JsToolMethods<MCPTool> for JsMCPTool {
    fn inner(&self) -> Arc<Mutex<MCPTool>> {
        self.inner.clone()
    }
}

#[napi]
impl JsMCPTool {
    #[napi(getter)]
    pub fn description(&self) -> napi::Result<ToolDesc> {
        self._description()
    }

    #[napi]
    pub async fn call(&self, kwargs: Option<Map<String, Value>>) -> napi::Result<Vec<JsPart>> {
        self._call(kwargs).await
    }
}

#[napi(js_name = "MCPTransport")]
pub struct JsMCPTransport(MCPTransport);

#[napi]
impl JsMCPTransport {
    #[napi(js_name = "type", getter)]
    pub fn transport_type(&self) -> String {
        match &self.0 {
            MCPTransport::Stdio { .. } => "Stdio".into(),
            MCPTransport::StreamableHttp { .. } => "StreamableHttp".into(),
        }
    }

    #[napi(getter, ts_return_type = "{ command: string, args: Array<string> }")]
    pub fn stdio(&self) -> napi::Result<Map<String, Value>> {
        match &self.0 {
            MCPTransport::Stdio { command, args } => {
                let mut map = Map::new();
                map.insert("command".into(), Value::String(command.clone()));
                map.insert(
                    "args".into(),
                    Value::Array(
                        args.into_iter()
                            .map(|arg| Value::String(arg.clone()))
                            .collect(),
                    ),
                );
                Ok(map)
            }
            _ => Err(napi::Error::new(
                Status::GenericFailure,
                "transport type is not Stdio",
            )),
        }
    }

    #[napi(getter, ts_return_type = "{ url: string }")]
    pub fn streamable_http(&self) -> napi::Result<Map<String, Value>> {
        match &self.0 {
            MCPTransport::StreamableHttp { url } => {
                let mut map = Map::new();
                map.insert("url".into(), Value::String(url.clone()));
                Ok(map)
            }
            _ => Err(napi::Error::new(
                Status::GenericFailure,
                "transport type is not StreamableHttp",
            )),
        }
    }

    #[napi]
    pub async fn tools(&self, tool_name_prefix: String) -> napi::Result<Vec<JsMCPTool>> {
        Ok(match self.0.clone() {
            MCPTransport::Stdio { command, args } => {
                mcp_tools_from_stdio(command.clone(), args.clone(), tool_name_prefix.as_str()).await
            }
            MCPTransport::StreamableHttp { url } => {
                mcp_tools_from_streamable_http(url.as_str(), tool_name_prefix.as_str()).await
            }
        }
        .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))?
        .into_iter()
        .map(|t| JsMCPTool {
            inner: Arc::new(Mutex::new(t)),
        })
        .collect::<Vec<JsMCPTool>>())
    }
}

#[napi]
pub struct JsFunctionTool {
    desc: ToolDesc,
    func: ThreadsafeFunction<Value, Promise<Value>, Value, Status, false, true>,
}

#[napi]
impl JsFunctionTool {
    #[napi(constructor)]
    pub fn new(
        desc: ToolDesc,
        func: ThreadsafeFunction<Value, Promise<Value>, Value, Status, false, true>,
    ) -> Self {
        Self { desc, func }
    }

    #[napi(getter)]
    pub fn description(&self) -> napi::Result<ToolDesc> {
        Ok(self.desc.clone())
    }

    #[napi]
    pub async fn call(&self, kwargs: Option<Map<String, Value>>) -> napi::Result<Vec<JsPart>> {
        let kwargs = serde_json::to_value(kwargs).unwrap();
        let promise = self.func.call_async(kwargs).await?;
        let result = match promise.await {
            Ok(result) => result.to_string(),
            // Failed tool call result should be also returned too
            Err(e) => e.to_string(),
        };
        let part = JsPart::new_text(serde_json::to_string(&result)?);
        Ok(vec![part])
    }
}
