// use std::borrow::Cow;
// use std::collections::HashMap;
// use std::sync::Arc;

// use async_trait::async_trait;
// use rmcp::model::*;

// #[cfg(not(target_arch = "wasm32"))]
// type PendingRequests =
//     Arc<tokio::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<JsonRpcResponse>>>>;
// #[cfg(not(target_arch = "wasm32"))]
// type PendingSender = tokio::sync::oneshot::Sender<JsonRpcResponse>;
// #[cfg(not(target_arch = "wasm32"))]
// type PendingReceiver = tokio::sync::oneshot::Receiver<JsonRpcResponse>;

// #[cfg(target_arch = "wasm32")]
// type PendingRequests =
//     Arc<futures::lock::Mutex<HashMap<String, futures::channel::oneshot::Sender<JsonRpcResponse>>>>;
// #[cfg(target_arch = "wasm32")]
// type PendingSender = futures::channel::oneshot::Sender<JsonRpcResponse>;
// #[cfg(target_arch = "wasm32")]
// type PendingReceiver = futures::channel::oneshot::Receiver<JsonRpcResponse>;

// // Error types
// #[derive(Debug, thiserror::Error)]
// pub enum McpError {
//     #[error("Transport error: {0}")]
//     Transport(String),
//     #[error("Serialization error: {0}")]
//     Serialization(#[from] serde_json::Error),
//     #[error("HTTP error: {0}")]
//     Http(String),
//     #[error("WebSocket error: {0}")]
//     WebSocket(String),
//     #[error("Connection closed")]
//     ConnectionClosed,
//     #[error("Initialization failed: {0}")]
//     InitializationFailed(String),
//     #[cfg(not(target_arch = "wasm32"))]
//     #[error("IO error: {0}")]
//     Io(#[from] std::io::Error),
//     #[cfg(not(target_arch = "wasm32"))]
//     #[error("Process error: {0}")]
//     Process(String),
// }

// fn create_channel() -> (PendingSender, PendingReceiver) {
//     #[cfg(not(target_arch = "wasm32"))]
//     {
//         tokio::sync::oneshot::channel()
//     }
//     #[cfg(target_arch = "wasm32")]
//     {
//         futures::channel::oneshot::channel()
//     }
// }

// fn spawn_future(fut: impl Future<Output = ()> + 'static) {
//     #[cfg(not(target_arch = "wasm32"))]
//     {
//         tokio::spawn(fut);
//     }

//     #[cfg(target_arch = "wasm32")]
//     {
//         wasm_bindgen_futures::spawn_local(fut);
//     }
// }

// #[cfg(not(target_arch = "wasm32"))]
// #[async_trait]
// pub trait Transport {
//     async fn send(
//         &self,
//         request: JsonRpcRequest<ClientRequest>,
//     ) -> Result<JsonRpcResponse, McpError>;
//     async fn close(&self) -> Result<(), McpError>;
// }

// #[cfg(target_arch = "wasm32")]
// #[async_trait(?Send)]
// pub trait Transport {
//     async fn send(
//         &self,
//         request: JsonRpcRequest<ClientRequest>,
//     ) -> Result<JsonRpcResponse, McpError>;
//     async fn close(&self) -> Result<(), McpError>;
// }

// macro_rules! impl_transport {
//     (
//         $struct_name:ident,
//         send: |$self_send:ident, $request:ident, $sender:ident, $receiver:ident, $request_id:ident| $send_body:block,
//         close: |$self_close:ident| $close_body:block
//     ) => {
//         // Generate Send implementation (non-wasm32)
//         #[cfg(not(target_arch = "wasm32"))]
//         #[async_trait]
//         impl Transport for $struct_name {
//             async fn send(
//                 &self,
//                 $request: JsonRpcRequest<ClientRequest>,
//             ) -> Result<JsonRpcResponse, McpError> {
//                 let ($sender, $receiver) = create_channel();
//                 let $request_id = $request.id.to_string();
//                 let $self_send = self;
//                 $send_body
//             }

//             async fn close(&self) -> Result<(), McpError> {
//                 let $self_close = self;
//                 $close_body
//             }
//         }

//         // Generate non-Send implementation (wasm32)
//         #[cfg(target_arch = "wasm32")]
//         #[async_trait(?Send)]
//         impl Transport for $struct_name {
//             async fn send(
//                 &self,
//                 $request: JsonRpcRequest<ClientRequest>,
//             ) -> Result<JsonRpcResponse, McpError> {
//                 let ($sender, $receiver) = create_channel();
//                 let $request_id = $request.id.to_string();
//                 let $self_send = self;
//                 $send_body
//             }

//             async fn close(&self) -> Result<(), McpError> {
//                 let $self_close = self;
//                 $close_body
//             }
//         }
//     };
// }

// pub struct StreamableHttpTransport {
//     base_url: String,
//     client: reqwest::Client,
//     pending_requests: PendingRequests,
// }

// impl StreamableHttpTransport {
//     pub fn new(base_url: impl Into<String>) -> Self {
//         Self {
//             base_url: base_url.into(),
//             client: reqwest::Client::new(),
//             pending_requests: Arc::new(Default::default()),
//         }
//     }

//     async fn start_http_stream(&self) -> Result<(), McpError> {
//         use futures::StreamExt;

//         let response = self
//             .client
//             .get(&self.base_url)
//             .header("Accept", "text/event-stream")
//             .send()
//             .await
//             .map_err(|e| McpError::Http(e.to_string()))?;

//         if !response.status().is_success() {
//             return Err(McpError::Http(format!(
//                 "Failed to establish stream: {}",
//                 response.status()
//             )));
//         }

//         let pending_requests = self.pending_requests.clone();
//         let mut stream = response.bytes_stream();

//         let stream_task = async move {
//             let mut buffer = String::new();

//             while let Some(chunk) = stream.next().await {
//                 match chunk {
//                     Ok(bytes) => {
//                         if let Ok(text) = String::from_utf8(bytes.to_vec()) {
//                             buffer.push_str(&text);

//                             while let Some(newline_pos) = buffer.find("\n") {
//                                 let buffer2 = buffer.clone();
//                                 let line = buffer2[..newline_pos].trim();
//                                 buffer = buffer[newline_pos + 1..].to_string();

//                                 if !line.is_empty() {
//                                     if let Ok(response) =
//                                         serde_json::from_str::<JsonRpcResponse>(line)
//                                     {
//                                         let id_str = response.id.to_string();
//                                         let mut pending = pending_requests.lock().await;
//                                         if let Some(sender) = pending.remove(&id_str) {
//                                             let _ = sender.send(response);
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                     Err(e) => {
//                         println!("HTTP stream error: {}", e);
//                         break;
//                     }
//                 }
//             }
//         };

//         spawn_future(stream_task);

//         Ok(())
//     }
// }

// impl_transport!(
//     StreamableHttpTransport,
//     send: |self, request, sender, receiver, request_id| {
//         {
//             let mut pending = self.pending_requests.lock().await;
//             pending.insert(request_id.clone(), sender);
//         }

//         let response = self
//             .client
//             .post(&self.base_url)
//             .header("Content-Type", "application/json")
//             .body(serde_json::to_value(&request)?.to_string())
//             .send()
//             .await
//             .map_err(|e| McpError::Http(e.to_string()))?;

//         if !response.status().is_success() {
//             let mut pending = self.pending_requests.lock().await;
//             pending.remove(&request_id);
//             return Err(McpError::Http(format!("HTTP error: {}", response.status())));
//         }

//         let json_response = receiver.await.map_err(|_| McpError::ConnectionClosed)?;
//         Ok(json_response)
//     },
//     close: |self| {
//         let mut pending = self.pending_requests.lock().await;
//         pending.clear();
//         Ok(())
//     }
// );

// pub struct McpClient {
//     transport: Box<dyn Transport>,
//     next_id: std::sync::atomic::AtomicU32,
// }

// impl McpClient {
//     fn new(transport: Box<dyn Transport>) -> Self {
//         Self {
//             transport,
//             next_id: std::sync::atomic::AtomicU32::new(1),
//         }
//     }

//     pub async fn with_streamable_http(base_url: impl Into<String>) -> Result<Self, McpError> {
//         let transport = StreamableHttpTransport::new(base_url);
//         transport.start_http_stream().await?;
//         Ok(Self::new(Box::new(transport)))
//     }

//     fn next_request_id(&self) -> NumberOrString {
//         NumberOrString::Number(
//             self.next_id
//                 .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
//         )
//     }

//     pub async fn initialize(
//         &self,
//         name: &str,
//         version: &str,
//     ) -> Result<InitializeResult, McpError> {
//         let request = JsonRpcRequest {
//             jsonrpc: Default::default(),
//             id: self.next_request_id(),
//             request: ClientRequest::InitializeRequest(InitializeRequest {
//                 method: Default::default(),
//                 params: InitializeRequestParam {
//                     protocol_version: Default::default(),
//                     capabilities: Default::default(),
//                     client_info: Implementation {
//                         name: name.to_string(),
//                         version: version.to_string(),
//                     },
//                 },
//                 extensions: Extensions::default(),
//             }),
//         };

//         let response = self.transport.send(request).await?;

//         let initialize_result: InitializeResult =
//             serde_json::from_value(serde_json::to_value(response.result)?).map_err(|e| {
//                 McpError::InitializationFailed(format!(
//                     "Invalid InitializeResult format: {}",
//                     e.to_string()
//                 ))
//             })?;

//         Ok(initialize_result)
//     }

//     pub async fn call_method(&self, request: ClientRequest) -> Result<JsonObject, McpError> {
//         let request = JsonRpcRequest {
//             request,
//             jsonrpc: Default::default(),
//             id: self.next_request_id(),
//         };
//         let response = self.transport.send(request).await?;
//         Ok(response.result)
//     }

//     pub async fn list_tools(&self) -> Result<ListToolsResult, McpError> {
//         let request = ClientRequest::ListToolsRequest(ListToolsRequest {
//             ..Default::default()
//         });
//         let resp = self.call_method(request).await?;
//         let resp: ListToolsResult = serde_json::from_value(serde_json::to_value(resp)?)?;
//         Ok(resp)
//     }

//     pub async fn call_tool(
//         &self,
//         name: Cow<'static, str>,
//         arguments: Option<JsonObject>,
//     ) -> Result<CallToolResult, McpError> {
//         let request = ClientRequest::CallToolRequest(CallToolRequest {
//             method: Default::default(),
//             params: CallToolRequestParam { name, arguments },
//             extensions: Default::default(),
//         });
//         let resp = self.call_method(request).await?;
//         let resp: CallToolResult = serde_json::from_value(serde_json::to_value(resp)?)?;
//         Ok(resp)
//     }

//     pub async fn close(&self) -> Result<(), McpError> {
//         self.transport.close().await
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use wasm_bindgen_test::*;

//     wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

//     #[wasm_bindgen_test]
//     async fn test_streamable_http() -> anyhow::Result<()> {
//         web_sys::console::log_1(&"sibal!".into());
//         let client = McpClient::with_streamable_http("http://localhost:8123/mcp").await?;
//         let result = client.initialize("test", "0.1.0").await;
//         assert!(result.is_ok(), "initialize failed: {:?}", result);
//         Ok(())
//     }
// }
