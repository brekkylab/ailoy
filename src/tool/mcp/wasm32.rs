use std::{borrow::Cow, rc::Rc, sync::atomic::AtomicU32};

use ailoy_macros::multi_platform_async_trait;
use anyhow::{Context, anyhow};
use futures::{StreamExt, stream::LocalBoxStream};
use rmcp::model::{
    CallToolRequest, CallToolRequestParam, CallToolResult, ClientJsonRpcMessage,
    ClientNotification, ClientRequest, InitializeRequest, InitializeRequestParam,
    InitializedNotification, JsonRpcResponse, ListToolsRequest, ListToolsResult, NumberOrString,
    ServerJsonRpcMessage, ServerResult,
};
use sse_stream::{Error as SseError, Sse, SseStream};
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{
    tool::{ToolBehavior, mcp::common::handle_result},
    value::{ToolDesc, Value},
};

type BoxedSseStream = LocalBoxStream<'static, Result<Sse, SseError>>;

enum StreamableHttpPostResponse {
    Accepted,
    Json(ServerJsonRpcMessage, Option<String>),
    Sse(BoxedSseStream, Option<String>),
}

impl std::fmt::Debug for StreamableHttpPostResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Accepted => write!(f, "Accepted"),
            Self::Json(arg0, arg1) => f.debug_tuple("Json").field(arg0).field(arg1).finish(),
            Self::Sse(_, arg1) => f.debug_tuple("Sse").field(arg1).finish(),
        }
    }
}

impl StreamableHttpPostResponse {
    pub async fn expect_initialized(
        self,
    ) -> Result<(ServerJsonRpcMessage, Option<String>), StreamableHttpError> {
        match self {
            Self::Json(message, session_id) => Ok((message, session_id)),
            Self::Sse(mut stream, session_id) => {
                let event =
                    stream
                        .next()
                        .await
                        .ok_or(StreamableHttpError::UnexpectedServerResponse(
                            "empty sse stream".into(),
                        ))??;
                let message: ServerJsonRpcMessage =
                    serde_json::from_str(&event.data.unwrap_or_default())?;
                Ok((message, session_id))
            }
            _ => Err(StreamableHttpError::UnexpectedServerResponse(
                "expect initialized, accepted".into(),
            )),
        }
    }

    pub fn expect_accepted(self) -> Result<(), StreamableHttpError> {
        match self {
            Self::Accepted => Ok(()),
            got => Err(StreamableHttpError::UnexpectedServerResponse(
                format!("expect accepted, got {got:?}").into(),
            )),
        }
    }

    pub async fn next(&mut self) -> Option<Result<ServerJsonRpcMessage, StreamableHttpError>> {
        match self {
            StreamableHttpPostResponse::Json(message, _) => Some(Ok(message.clone())),
            StreamableHttpPostResponse::Sse(stream, _) => loop {
                match stream.next().await {
                    Some(Ok(sse)) => {
                        if let Some(data) = sse.data {
                            match serde_json::from_str::<ServerJsonRpcMessage>(&data) {
                                Ok(message) => {
                                    return Some(Ok(message));
                                }
                                Err(e) => {
                                    return Some(Err(
                                        StreamableHttpError::UnexpectedServerResponse(
                                            format!("{e}").into(),
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                    Some(Err(_)) => {
                        return Some(Err(StreamableHttpError::UnexpectedEndOfStream));
                    }
                    None => {
                        return None;
                    }
                }
            },
            StreamableHttpPostResponse::Accepted => None,
        }
    }
}

#[derive(Error, Debug)]
enum StreamableHttpError {
    #[error("SSE error: {0}")]
    Sse(#[from] SseError),
    #[error("Io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Client error: {0}")]
    Client(#[from] reqwest::Error),
    #[error("unexpected end of stream")]
    UnexpectedEndOfStream,
    #[error("unexpected server response: {0}")]
    UnexpectedServerResponse(Cow<'static, str>),
    #[error("Unexpected content type: {0:?}")]
    UnexpectedContentType(Option<String>),
    #[error("Deserialize error: {0}")]
    Deserialize(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
struct StreamableHttpClient {
    inner: reqwest::Client,
    url: url::Url,
    auth_token: Option<String>,
    session_id: Option<String>,
    allow_stateless: bool,
    request_id: Rc<AtomicU32>,
}

impl StreamableHttpClient {
    pub fn new(url: &str, auth_token: Option<String>, allow_stateless: bool) -> Self {
        let url = url::Url::parse(url).unwrap();
        Self {
            inner: reqwest::Client::new(),
            url: url,
            auth_token,
            session_id: None,
            allow_stateless,
            request_id: Rc::new(AtomicU32::new(0)),
        }
    }

    fn next_request_id(&self) -> NumberOrString {
        NumberOrString::Number(
            self.request_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        )
    }

    async fn post_message(
        &self,
        message: ClientJsonRpcMessage,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError> {
        let mut request = self.inner.post(self.url.as_ref()).header(
            "Accept",
            ["text/event-stream", "application/json"].join(", "),
        );
        if let Some(auth_header) = self.auth_token.clone() {
            request = request.bearer_auth(auth_header);
        }
        if self.session_id.is_some() {
            request = request.header("Mcp-Session-Id", self.session_id.as_ref().unwrap());
        }
        let response = request.json(&message).send().await?.error_for_status()?;
        if response.status() == reqwest::StatusCode::ACCEPTED {
            return Ok(StreamableHttpPostResponse::Accepted);
        }

        let content_type = response.headers().get(reqwest::header::CONTENT_TYPE);
        let session_id = response.headers().get("Mcp-Session-Id");
        let session_id = session_id
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        match content_type {
            Some(ct) if ct.as_bytes().starts_with("text/event-stream".as_bytes()) => {
                let event_stream =
                    SseStream::from_byte_stream(response.bytes_stream()).boxed_local();
                Ok(StreamableHttpPostResponse::Sse(event_stream, session_id))
            }
            Some(ct) if ct.as_bytes().starts_with("application/json".as_bytes()) => {
                let message: ServerJsonRpcMessage = response.json().await?;
                Ok(StreamableHttpPostResponse::Json(message, session_id))
            }
            _ => Err(StreamableHttpError::UnexpectedContentType(
                content_type.map(|ct| String::from_utf8_lossy(ct.as_bytes()).to_string()),
            )),
        }
    }

    async fn initialize(&mut self) -> anyhow::Result<()> {
        let initialize_request = ClientJsonRpcMessage::request(
            ClientRequest::InitializeRequest(InitializeRequest {
                method: Default::default(),
                params: InitializeRequestParam {
                    protocol_version: Default::default(),
                    capabilities: Default::default(),
                    client_info: Default::default(),
                },
                extensions: Default::default(),
            }),
            self.next_request_id(),
        );
        let (message, session_id) = self
            .post_message(initialize_request)
            .await?
            .expect_initialized()
            .await?;

        // The session id should be exist if stateless is not allowed
        if let Some(session_id) = session_id.clone() {
            self.session_id = Some(session_id.clone());
        } else if !self.allow_stateless {
            return Err(anyhow!("Missing session id in initialize response"));
        }

        // The arrived message should be matched with InitializeResult
        match message {
            ServerJsonRpcMessage::Response(JsonRpcResponse {
                result: ServerResult::InitializeResult(..),
                ..
            }) => {}
            _ => {
                return Err(anyhow!(
                    "expected initialized response, but received: {0:?}",
                    message
                ));
            }
        }

        // Send initialized notification
        let notification = ClientJsonRpcMessage::notification(
            ClientNotification::InitializedNotification(InitializedNotification {
                method: Default::default(),
                extensions: Default::default(),
            }),
        );
        self.post_message(notification)
            .await
            .context("Failed to send initialized notification")?
            .expect_accepted()
            .context("Response of initialized notification is not accepted")?;

        Ok(())
    }

    async fn send_request(&self, request: ClientRequest) -> anyhow::Result<ServerJsonRpcMessage> {
        let request_id = self.next_request_id();
        let message = ClientJsonRpcMessage::request(request, request_id);
        let mut response = self.post_message(message).await?;

        while let Some(result) = response.next().await {
            let message = result?;
            match &message {
                ServerJsonRpcMessage::Response(_) | ServerJsonRpcMessage::Error(_) => {
                    return Ok(message);
                }
                ServerJsonRpcMessage::Notification(_) => {
                    continue;
                }
                _ => {
                    continue;
                }
            }
        }
        Err(anyhow!("No server response"))
    }

    pub async fn list_tools(&self) -> anyhow::Result<ListToolsResult> {
        let request = ClientRequest::ListToolsRequest(ListToolsRequest {
            method: Default::default(),
            params: Default::default(),
            extensions: Default::default(),
        });
        let response = self.send_request(request).await?;

        match response {
            ServerJsonRpcMessage::Response(JsonRpcResponse {
                result: ServerResult::ListToolsResult(list_tools_result),
                ..
            }) => Ok(list_tools_result),
            _ => Err(anyhow!(
                "expected ListToolsResult response, but received: {0:?}",
                response
            )),
        }
    }

    pub async fn call_tool(&self, params: CallToolRequestParam) -> anyhow::Result<CallToolResult> {
        let request = ClientRequest::CallToolRequest(CallToolRequest {
            method: Default::default(),
            params,
            extensions: Default::default(),
        });
        let response = self.send_request(request).await?;

        match response {
            ServerJsonRpcMessage::Response(JsonRpcResponse {
                result: ServerResult::CallToolResult(call_tool_result),
                ..
            }) => Ok(call_tool_result),
            _ => Err(anyhow!(
                "expected CallToolResult response, but received: {0:?}",
                response
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[wasm_bindgen(getter_with_clone)]
pub struct MCPClient {
    /// In wasm, only the streamable http transport is allowed
    _client: Rc<StreamableHttpClient>,
    #[wasm_bindgen(skip)]
    pub tools: Vec<MCPTool>,
}

impl MCPClient {
    pub async fn from_streamable_http(url: impl Into<String>) -> anyhow::Result<Self> {
        let mut client = StreamableHttpClient::new(&url.into(), None, true);
        client.initialize().await?;

        let tools = client
            .list_tools()
            .await?
            .tools
            .into_iter()
            .map(|t| MCPTool {
                client: Rc::new(client.clone()),
                inner: t.clone(),
            })
            .collect();

        Ok(Self {
            _client: Rc::new(client),
            tools,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MCPTool {
    client: Rc<StreamableHttpClient>,
    inner: rmcp::model::Tool,
}

#[multi_platform_async_trait]
impl ToolBehavior for MCPTool {
    fn get_description(&self) -> ToolDesc {
        ToolDesc {
            name: self.inner.name.to_string(),
            description: self.inner.description.clone().map(|v| v.into()),
            parameters: self
                .inner
                .input_schema
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        <serde_json::Value as Into<Value>>::into(v.clone()),
                    )
                })
                .collect(),
            returns: self.inner.output_schema.clone().map(|map| {
                map.iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            <serde_json::Value as Into<Value>>::into(v.clone()),
                        )
                    })
                    .collect()
            }),
        }
    }

    async fn run(&self, args: Value) -> anyhow::Result<Value> {
        let client = self.client.clone();
        let tool_name = self.inner.name.clone();

        // Convert your ToolCall arguments → serde_json::Map (MCP expects JSON object)
        let arguments: Option<serde_json::Map<String, serde_json::Value>> =
            serde_json::to_value(args)
                .context("serialize ToolCall arguments failed: {e}")?
                .as_object()
                .cloned();

        let result = client
            .call_tool(CallToolRequestParam {
                name: tool_name.into(),
                arguments,
            })
            .await
            .context("mcp call_tool failed: {e}")?;

        let parts = handle_result(result).context("call_tool_result_to_parts failed")?;
        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    use super::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_streamable_http_client() -> anyhow::Result<()> {
        let mut client = StreamableHttpClient::new("http://localhost:8123/mcp", None, true);
        client.initialize().await?;

        let list_tools = client.list_tools().await?;
        crate::debug!("list of tools: {:?}", list_tools);

        let call_tool = client
            .call_tool(CallToolRequestParam {
                name: "get-forecast".into(),
                arguments: Some(
                    serde_json::json!({"latitude": 32.7767, "longitude": -96.797})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            })
            .await
            .unwrap();
        crate::debug!("call tool result: {:?}", call_tool);

        Ok(())
    }

    #[wasm_bindgen_test]
    async fn test_mcp_tools_from_streamable_http() -> anyhow::Result<()> {
        let client = MCPClient::from_streamable_http("http://localhost:8123/mcp").await?;
        let tool = client.tools[1].clone();

        let args = serde_json::from_value::<Value>(
            serde_json::json!({"latitude": 32.7767, "longitude": -96.797}),
        )?;
        let result = tool.run(args).await.unwrap();
        crate::debug!("tool call result: {:?}", result);

        Ok(())
    }
}
