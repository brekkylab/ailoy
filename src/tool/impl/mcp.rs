//! MCP (Model Context Protocol) client support.
//!
//! An MCP server is not one tool but a bag of them, and the list is knowable
//! only by asking: the client connects, initialises, and calls `tools/list`.
//! That is a round trip — over a socket, or to a child process — and
//! [`ToolProvider::provide`](crate::tool::ToolProvider::provide) is a `fn` with
//! no `.await` to spend on it. So discovery happens once, at **registration**
//! time, and what lands in the registry is one already-resolved entry per remote
//! tool, each sharing the live connection behind an `Arc`:
//!
//! ```text
//! MCPConnection::connect(&transport)   ← async: initialize + tools/list
//!         │
//!         │  ToolProvider::insert_mcp("github", conn)   ← sync: N entries
//!         ▼
//! "github__create_issue" → ToolProviderElem::MCP(MCPToolEntry)
//! "github__list_issues"  → ToolProviderElem::MCP(MCPToolEntry)
//! ```
//!
//! Registration also hands back the [`ToolDesc`]s it just inserted, which is
//! what an [`AgentSpec`](crate::agent::AgentSpec) wants in its `tools` field —
//! so the spec stays a plain list of descriptions and nothing downstream of
//! registration needs to know an MCP server was involved.
//!
//! ```no_run
//! # use ailoy::{agent::AgentSpec, tool::{register_mcp_stdio, unregister_mcp}};
//! # async fn example() -> anyhow::Result<()> {
//! let descs = register_mcp_stdio(
//!     "default",
//!     "github",
//!     "npx",
//!     ["-y", "@modelcontextprotocol/server-github"],
//! )
//! .await?;
//!
//! let spec = AgentSpec::new("anthropic/claude-sonnet-4-6")
//!     .system_tools()
//!     .tools(descs);
//!
//! // ... build and run agents against `spec` ...
//!
//! // Ends the child process; without this it outlives every agent that used it.
//! unregister_mcp("default", "github")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Names
//!
//! The name a model sees is `{prefix}__{remote name}`, because two servers may
//! both call a tool `search` and the registry is one flat name-keyed map. The
//! separator is `__` rather than `/` or `.`: model-facing tool names are limited
//! to `[A-Za-z0-9_-]` by the OpenAI and Anthropic schemas, so a `/` would make
//! the request itself invalid. Doubling the underscore keeps the boundary
//! legible when a prefix or a remote name contains one of its own.
//!
//! The remote name is kept beside the entry and used verbatim on the wire — the
//! prefix is this crate's business, not the server's.
//!
//! ## Where a stdio server runs
//!
//! On the **host**, not inside the [`Console`](crate::console::Console) sandbox
//! that the built-in tools run in. [`Console::exec`](cortex::console::Console::exec)
//! is argv-in, bytes-out — one shot, with no handle to a process left running —
//! so there is nowhere inside the sandbox to keep a server that has to hold its
//! stdin and stdout open for the length of a session. An MCP server therefore
//! has whatever access this process has. That is the one asymmetry between MCP
//! tools and every other tool here, and the reason to register only servers the
//! caller trusts.

use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rmcp::{
    RoleClient, ServiceExt as _,
    model::{CallToolRequestParams, CallToolResult, ContentBlock, ResourceContents},
    service::RunningService,
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};

use crate::{
    datatype::{Bytes, Value},
    message::{Message, Part, Role},
    tool::{MCPToolProviderElem, ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

/// Separator between the registry prefix and the server's own tool name.
///
/// See the module docs on names for why it is not `/`.
pub const MCP_NAME_SEPARATOR: &str = "__";

/// Longest tool name the stricter of the two model APIs (OpenAI) accepts.
/// Exceeding it is warned about rather than refused: it is the model provider's
/// limit to enforce, and which provider this agent uses is not known here.
const MAX_TOOL_NAME_LEN: usize = 64;

// ── Connection ────────────────────────────────────────────────────────────────

/// A live MCP session plus the tool list it reported at startup.
///
/// Created by [`MCPToolProviderElem::connect`] and handed to
/// [`ToolProvider::insert_mcp`](crate::tool::ToolProvider::insert_mcp), which
/// wraps it in an `Arc` and fans it out into one registry entry per tool.
pub struct MCPConnection {
    service: RunningService<RoleClient, ()>,
    tools: Vec<rmcp::model::Tool>,
}

impl std::fmt::Debug for MCPConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MCPConnection")
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|t| t.name.as_ref())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl MCPConnection {
    /// Spawn `command` and speak MCP over its stdio.
    ///
    /// The child's stderr goes to `/dev/null`: servers routinely log there, and
    /// inheriting it would interleave that with whatever the host process is
    /// drawing on its own terminal.
    pub async fn stdio(
        command: impl AsRef<str>,
        args: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Self> {
        let mut cmd = tokio::process::Command::new(command.as_ref());
        cmd.args(args.into_iter().map(|a| a.as_ref().to_string()));

        let (transport, _stderr) = TokioChildProcess::builder(cmd)
            .stderr(std::process::Stdio::null())
            .spawn()?;

        Self::serve(transport).await
    }

    /// Connect to a remote MCP server over streamable HTTP.
    pub async fn streamable_http(url: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::serve(StreamableHttpClientTransport::from_uri(
            url.as_ref().to_string(),
        ))
        .await
    }

    /// Drive `initialize` and the paginated `tools/list` that follows it.
    ///
    /// `()` is rmcp's do-nothing client handler: this crate is a tool caller, so
    /// it answers no server-initiated requests (no sampling, no roots).
    pub(crate) async fn serve<T, E, A>(transport: T) -> anyhow::Result<Self>
    where
        T: rmcp::transport::IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = ().serve(transport).await?;
        let tools = service.peer().list_all_tools().await?;
        Ok(Self { service, tools })
    }

    /// The tools the server reported, as it named them.
    pub fn tools(&self) -> &[rmcp::model::Tool] {
        &self.tools
    }

    /// Close the session, which for a stdio server also ends the child process.
    ///
    /// Cancellation rather than [`RunningService::cancel`] because the entries in
    /// the registry share this connection through an `Arc` and so none of them
    /// can consume it by value.
    pub fn shutdown(&self) {
        self.service.cancellation_token().cancel();
    }
}

impl MCPToolProviderElem {
    /// Open a session against the server this transport describes.
    pub async fn connect(&self) -> anyhow::Result<MCPConnection> {
        match self {
            MCPToolProviderElem::Stdio { command, args } => {
                MCPConnection::stdio(command, args).await
            }
            MCPToolProviderElem::StreamableHTTP { url } => {
                MCPConnection::streamable_http(url.as_str()).await
            }
        }
    }
}

// ── Registry entry ────────────────────────────────────────────────────────────

/// One remote tool, bound to the connection that serves it.
///
/// Cloning shares the session; every entry from the same server points at the
/// same `Arc`, which is what makes shutting one down a single operation.
#[derive(Clone)]
pub struct MCPToolEntry {
    conn: Arc<MCPConnection>,
    /// The name the server knows this tool by — what goes on the wire, before
    /// the registry prefix was put in front of it.
    remote_name: String,
}

impl std::fmt::Debug for MCPToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MCPToolEntry")
            .field("remote_name", &self.remote_name)
            .finish()
    }
}

impl MCPToolEntry {
    pub(crate) fn new(conn: Arc<MCPConnection>, remote_name: impl Into<String>) -> Self {
        Self {
            conn,
            remote_name: remote_name.into(),
        }
    }

    /// The connection this entry calls through, for identity comparisons when
    /// removing a whole server from a registry.
    pub(crate) fn conn(&self) -> &Arc<MCPConnection> {
        &self.conn
    }

    /// Build the [`ToolFunc`] that performs `tools/call` for this tool.
    ///
    /// Pure, not console-bound: the call goes to the server over a connection
    /// this entry already holds, so there is no sandbox to borrow and the
    /// resulting stream is `'static`.
    pub(crate) fn tool_func(&self) -> ToolFunc {
        let conn = self.conn.clone();
        let remote_name = self.remote_name.clone();

        tool_func!(async |args: Value, id: String| -> Message
            with [conn = conn.clone(), remote_name = remote_name.clone()]
            {
            // MCP takes an arguments *object*; anything else is a tool being
            // called with something its schema never described, and saying so
            // is more use to the model than an empty call the server rejects.
            let arguments = match serde_json::Value::from(args) {
                serde_json::Value::Object(map) => Some(map),
                serde_json::Value::Null => None,
                other => {
                    return error_message(
                        id,
                        format!(
                            "expected a JSON object of arguments, got {}",
                            type_name_of(&other)
                        ),
                    );
                }
            };

            let mut params = CallToolRequestParams::new(remote_name.clone());
            if let Some(arguments) = arguments {
                params = params.with_arguments(arguments);
            }

            match conn.service.peer().call_tool(params).await {
                Ok(result) => Message::new(Role::Tool)
                    .with_contents(call_tool_result_to_parts(result))
                    .with_id(id),
                Err(e) => error_message(id, e.to_string()),
            }
        })
    }
}

fn error_message(id: String, detail: impl std::fmt::Display) -> Message {
    Message::new(Role::Tool)
        .with_contents([Part::text(format!("Error: {detail}"))])
        .with_id(id)
}

fn type_name_of(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

// ── Descriptions ──────────────────────────────────────────────────────────────

/// Turn a server's tool into the [`ToolDesc`] a spec carries, under `prefix`.
pub(crate) fn mcp_tool_desc(prefix: &str, tool: &rmcp::model::Tool) -> ToolDesc {
    let name = prefixed_tool_name(prefix, &tool.name);

    if name.len() > MAX_TOOL_NAME_LEN {
        log::warn!(
            "MCP tool name '{name}' is {} characters; some model APIs reject names over {MAX_TOOL_NAME_LEN}",
            name.len()
        );
    }

    // The schema is passed through as the server wrote it. It is already JSON
    // Schema, which is what `ToolDesc::parameters` holds, and rewriting it here
    // would only risk disagreeing with the server about what it accepts.
    let parameters = Value::from(serde_json::Value::Object((*tool.input_schema).clone()));

    let mut builder = ToolDescBuilder::new(name).parameters(parameters);
    if let Some(description) = tool.description.as_ref() {
        builder = builder.description(description.to_string());
    }
    if let Some(output_schema) = tool.output_schema.as_ref() {
        builder = builder.returns(Value::from(serde_json::Value::Object(
            (**output_schema).clone(),
        )));
    }
    builder.build()
}

/// `{prefix}__{name}`, with anything a model API would refuse mapped to `_`.
///
/// Sanitising rather than refusing: the remote name is kept verbatim on the
/// entry and used on the wire, so a renamed tool still calls the right thing,
/// and one oddly-named tool out of forty should not cost the caller the server.
pub(crate) fn prefixed_tool_name(prefix: &str, remote_name: &str) -> String {
    let mut out =
        String::with_capacity(prefix.len() + MCP_NAME_SEPARATOR.len() + remote_name.len());
    out.push_str(&sanitize_name_part(prefix));
    out.push_str(MCP_NAME_SEPARATOR);
    out.push_str(&sanitize_name_part(remote_name));
    out
}

fn sanitize_name_part(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ── Result mapping ────────────────────────────────────────────────────────────

/// Map a `tools/call` result onto the parts of a tool message.
///
/// A server that declares an `outputSchema` returns both `structuredContent` and
/// a text rendering of it in `content`; the structured form is preferred, since
/// it is the one the schema describes and the one a caller can parse.
pub(crate) fn call_tool_result_to_parts(result: CallToolResult) -> Vec<Part> {
    if result.is_error.unwrap_or(false) {
        // Errors are the server's own report of a failed call, not a transport
        // failure, so they belong in the conversation for the model to react to.
        let detail = result
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let detail = if detail.is_empty() {
            "the tool reported an error with no detail".to_string()
        } else {
            detail
        };
        return vec![Part::text(format!("Error: {detail}"))];
    }

    if let Some(structured) = result.structured_content {
        return vec![Part::value(Value::from(structured))];
    }

    let mut parts = Vec::with_capacity(result.content.len());
    for block in result.content {
        parts.push(content_block_to_part(block));
    }
    if parts.is_empty() {
        // A call that returned nothing still has to say so: an empty content
        // list would otherwise reach the model as a tool message with no parts.
        parts.push(Part::value(Value::Null));
    }
    parts
}

fn content_block_to_part(block: ContentBlock) -> Part {
    match block {
        ContentBlock::Text(text) => Part::text(text.text),

        ContentBlock::Image(image) => match BASE64.decode(image.data.as_bytes()) {
            Ok(data) => Part::image_embedded(image.mime_type.clone(), Bytes::from(data))
                .unwrap_or_else(|e| Part::text(format!("[unreadable image: {e}]"))),
            Err(e) => Part::text(format!("[image with undecodable base64 payload: {e}]")),
        },

        // No audio part exists in this crate's message model, and inlining the
        // base64 would put a blob in the history that no model reads. Describe
        // it instead, so the model knows something came back and what it was.
        ContentBlock::Audio(audio) => Part::text(format!(
            "[audio omitted: {}, {} bytes base64]",
            audio.mime_type,
            audio.data.len()
        )),

        ContentBlock::Resource(resource) => match resource.resource {
            ResourceContents::TextResourceContents { uri, text, .. } => {
                Part::text(format!("{uri}:\n{text}"))
            }
            ResourceContents::BlobResourceContents { uri, mime_type, .. } => Part::text(format!(
                "[binary resource omitted: {uri}{}]",
                mime_type.map(|m| format!(", {m}")).unwrap_or_default()
            )),
            // `#[non_exhaustive]`, like `ContentBlock` below: a newer protocol
            // can carry a resource shape this build has no arm for.
            other => Part::text(format!("[unsupported MCP resource: {other:?}]")),
        },

        ContentBlock::ResourceLink(link) => {
            Part::text(format!("[resource link: {} ({})]", link.name, link.uri))
        }

        // `ContentBlock` is `#[non_exhaustive]`: a server speaking a newer
        // protocol than this build can send a kind that did not exist when it
        // was compiled, and dropping it silently would look like an empty result.
        other => Part::text(format!("[unsupported MCP content block: {other:?}]")),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;
    use rmcp::{
        ServerHandler,
        model::{
            CallToolResponse, ErrorData as McpError, ListToolsResult, PaginatedRequestParams,
            ServerInfo, Tool,
        },
        service::RequestContext,
    };

    use super::*;
    use crate::tool::ToolProvider;

    // ── A server to talk to ───────────────────────────────────────────────────

    /// Two tools, enough to cover the shapes the mapping has to distinguish:
    /// `echo` answers with plain content, `structured` with `structuredContent`,
    /// and either reports an error when asked to.
    #[derive(Clone)]
    struct TestServer;

    fn schema(json: serde_json::Value) -> Arc<rmcp::model::JsonObject> {
        Arc::new(json.as_object().expect("schema must be an object").clone())
    }

    impl ServerHandler for TestServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::default()
        }

        fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<rmcp::RoleServer>,
        ) -> impl Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
            let tools = vec![
                Tool::new(
                    "echo",
                    "Echo the message back",
                    schema(serde_json::json!({
                        "type": "object",
                        "properties": { "message": { "type": "string" } },
                        "required": ["message"]
                    })),
                ),
                // A name with characters the model APIs refuse, to prove the
                // sanitised name is what gets registered and the raw one is
                // still what goes on the wire.
                Tool::new(
                    "odd.name/tool",
                    "Returns structured content",
                    schema(serde_json::json!({ "type": "object" })),
                ),
            ];
            std::future::ready(Ok(ListToolsResult::with_all_items(tools)))
        }

        fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<rmcp::RoleServer>,
        ) -> impl Future<Output = Result<CallToolResponse, McpError>> + Send + '_ {
            let result = match request.name.as_ref() {
                "echo" => {
                    let message = request
                        .arguments
                        .as_ref()
                        .and_then(|a| a.get("message"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    if message == "boom" {
                        CallToolResult::error(vec![ContentBlock::text("it went boom")])
                    } else {
                        CallToolResult::success(vec![ContentBlock::text(message)])
                    }
                }
                "odd.name/tool" => {
                    // Both halves, as a server with an `outputSchema` really
                    // answers: the structured value and a text rendering of it.
                    let mut result =
                        CallToolResult::success(vec![ContentBlock::text("{\"ok\":true}")]);
                    result.structured_content = Some(serde_json::json!({ "ok": true }));
                    result
                }
                other => {
                    return std::future::ready(Err(McpError::invalid_params(
                        format!("no such tool: {other}"),
                        None,
                    )));
                }
            };
            std::future::ready(Ok(CallToolResponse::Complete(result)))
        }
    }

    /// Client and server over an in-memory pipe — the real protocol, including
    /// `initialize` and `tools/list`, without a process or a socket.
    async fn connected() -> MCPConnection {
        let (client_io, server_io) = tokio::io::duplex(8192);
        tokio::spawn(async move {
            if let Ok(server) = TestServer.serve(server_io).await {
                let _ = server.waiting().await;
            }
        });
        MCPConnection::serve(client_io)
            .await
            .expect("connecting to the test server")
    }

    // ── Names ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_prefixed_name_uses_double_underscore() {
        assert_eq!(
            prefixed_tool_name("github", "create_issue"),
            "github__create_issue"
        );
    }

    #[test]
    fn test_prefixed_name_sanitizes_both_halves() {
        // `/` and `.` are refused by the OpenAI and Anthropic name schemas.
        assert_eq!(
            prefixed_tool_name("my.server", "odd/name"),
            "my_server__odd_name"
        );
    }

    #[test]
    fn test_prefixed_name_keeps_hyphens_and_digits() {
        assert_eq!(prefixed_tool_name("srv-1", "get-time2"), "srv-1__get-time2");
    }

    // ── Descriptions ──────────────────────────────────────────────────────────

    #[test]
    fn test_tool_desc_passes_schema_through_unchanged() {
        let input = serde_json::json!({
            "type": "object",
            "properties": { "message": { "type": "string" } },
            "required": ["message"]
        });
        let tool = Tool::new("echo", "Echo it", schema(input.clone()));

        let desc = mcp_tool_desc("srv", &tool);

        assert_eq!(desc.name, "srv__echo");
        assert_eq!(desc.description.as_deref(), Some("Echo it"));
        assert_eq!(serde_json::Value::from(desc.parameters), input);
        assert!(desc.returns.is_none());
    }

    // ── Result mapping ────────────────────────────────────────────────────────

    #[test]
    fn test_structured_content_wins_over_its_text_rendering() {
        let mut result = CallToolResult::success(vec![ContentBlock::text("{\"ok\":true}")]);
        result.structured_content = Some(serde_json::json!({ "ok": true }));
        let parts = call_tool_result_to_parts(result);
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0]
                .as_value()
                .map(|v| serde_json::Value::from(v.clone())),
            Some(serde_json::json!({ "ok": true }))
        );
    }

    #[test]
    fn test_error_result_becomes_text_the_model_can_read() {
        let result = CallToolResult::error(vec![ContentBlock::text("rate limited")]);
        let parts = call_tool_result_to_parts(result);
        assert_eq!(parts[0].as_text(), Some("Error: rate limited"));
    }

    #[test]
    fn test_empty_result_still_produces_a_part() {
        let parts = call_tool_result_to_parts(CallToolResult::default());
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].as_value(), Some(&Value::Null));
    }

    #[test]
    fn test_image_content_is_decoded_not_dropped() {
        let png = vec![0x89, 0x50, 0x4E, 0x47];
        let result =
            CallToolResult::success(vec![ContentBlock::image(BASE64.encode(&png), "image/png")]);
        let parts = call_tool_result_to_parts(result);
        assert!(
            parts[0].is_image(),
            "expected an image part, got {:?}",
            parts[0]
        );
    }

    #[test]
    fn test_undecodable_image_degrades_to_text() {
        let result =
            CallToolResult::success(vec![ContentBlock::image("not base64!!", "image/png")]);
        let parts = call_tool_result_to_parts(result);
        assert!(parts[0].as_text().unwrap().contains("undecodable"));
    }

    // ── End to end ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_discovery_reports_every_tool() {
        let conn = connected().await;
        let names: Vec<_> = conn.tools().iter().map(|t| t.name.to_string()).collect();
        assert_eq!(names, vec!["echo", "odd.name/tool"]);
    }

    #[tokio::test]
    async fn test_registration_fans_one_server_out_into_entries() {
        let conn = connected().await;
        let mut provider = ToolProvider::empty();
        let descs = provider.insert_mcp("test", conn);

        let names: Vec<_> = descs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["test__echo", "test__odd_name_tool"]);
        assert!(provider.get("test__echo").is_some());
    }

    #[tokio::test]
    async fn test_provided_func_calls_through_to_the_server() {
        let conn = connected().await;
        let mut provider = ToolProvider::empty();
        let descs = provider.insert_mcp("test", conn);

        let funcs = provider.provide(&descs).expect("providing MCP descs");
        let func = funcs.get("test__echo").expect("echo was registered");

        // Pure: an MCP call needs no console, so the stream is `'static`.
        let out = func
            .call_pure(crate::to_value!({ "message": "hello" }), "call-1")
            .expect("MCP tools are pure")
            .next()
            .await
            .expect("one output");

        assert_eq!(out.message.contents[0].as_text(), Some("hello"));
        assert_eq!(out.message.id.as_deref(), Some("call-1"));
    }

    #[tokio::test]
    async fn test_sanitized_name_still_calls_the_servers_own_name() {
        let conn = connected().await;
        let mut provider = ToolProvider::empty();
        let descs = provider.insert_mcp("test", conn);

        let funcs = provider.provide(&descs).unwrap();
        let out = funcs
            .get("test__odd_name_tool")
            .expect("the renamed tool was registered")
            .call_pure(Value::object_empty(), "call-1")
            .unwrap()
            .next()
            .await
            .unwrap();

        // Reaching the handler at all proves `odd.name/tool` went out on the
        // wire, not the sanitised `odd_name_tool` the model sees.
        assert_eq!(
            out.message.contents[0]
                .as_value()
                .map(|v| serde_json::Value::from(v.clone())),
            Some(serde_json::json!({ "ok": true }))
        );
    }

    #[tokio::test]
    async fn test_server_side_error_reaches_the_model_as_text() {
        let conn = connected().await;
        let mut provider = ToolProvider::empty();
        let descs = provider.insert_mcp("test", conn);

        let funcs = provider.provide(&descs).unwrap();
        let out = funcs
            .get("test__echo")
            .unwrap()
            .call_pure(crate::to_value!({ "message": "boom" }), "call-1")
            .unwrap()
            .next()
            .await
            .unwrap();

        assert_eq!(
            out.message.contents[0].as_text(),
            Some("Error: it went boom")
        );
    }

    #[tokio::test]
    async fn test_non_object_arguments_are_reported_not_sent() {
        let conn = connected().await;
        let mut provider = ToolProvider::empty();
        let descs = provider.insert_mcp("test", conn);

        let funcs = provider.provide(&descs).unwrap();
        let out = funcs
            .get("test__echo")
            .unwrap()
            .call_pure(Value::string("just a string"), "call-1")
            .unwrap()
            .next()
            .await
            .unwrap();

        let text = out.message.contents[0].as_text().unwrap();
        assert!(text.starts_with("Error:"), "got {text}");
        assert!(text.contains("a string"), "got {text}");
    }

    #[tokio::test]
    async fn test_remove_mcp_takes_the_whole_server() {
        let conn = connected().await;
        let mut provider = ToolProvider::empty();
        provider.insert_mcp("test", conn);

        assert_eq!(provider.remove_mcp("test"), 2);
        assert!(provider.get("test__echo").is_none());
        assert_eq!(provider.remove_mcp("test"), 0);
    }

    #[tokio::test]
    async fn test_remove_mcp_does_not_take_a_neighbouring_prefix() {
        let mut provider = ToolProvider::empty();
        provider.insert_mcp("test", connected().await);
        provider.insert_mcp("test2", connected().await);

        // "test" must not swallow "test2": the match includes the separator.
        assert_eq!(provider.remove_mcp("test"), 2);
        assert!(provider.get("test2__echo").is_some());
    }
}
