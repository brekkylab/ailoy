use std::{collections::HashMap, pin::Pin, sync::Arc};

use futures::{Stream, future::BoxFuture};
use tokio::sync::Mutex as TokioMutex;

use crate::{
    agent::{AgentRuntime, BuiltinToolProvider, MCPToolProvider},
    datatype::Value,
    message::{Message, Part, Role, StreamingToolOutput, ToolDesc},
};

mod subagent;
mod web_search;

/// A synchronous tool function that returns a [`Value`] directly.
pub type ToolSyncFunc = dyn Fn(Value) -> Value + Send + Sync;

/// An asynchronous tool function that returns a [`Value`] via a future.
pub type ToolAsyncFunc = dyn Fn(Value) -> BoxFuture<'static, Value> + Send + Sync;

/// A streaming tool function that yields incremental [`ToolResultDelta`] items.
///
/// The stream must yield zero or more [`StreamingToolOutput::Delta`] items
/// (intermediate progress for UI display), followed by exactly one
/// [`StreamingToolOutput::Result`] item (the definitive tool output).
pub type ToolStreamingFunc = dyn Fn(Value) -> BoxFuture<'static, Pin<Box<dyn Stream<Item = StreamingToolOutput> + Send>>>
    + Send
    + Sync;

/// Identifies the origin of a [`ToolRuntime`].
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum ToolKind {
    /// Built into the agent runtime (e.g. web search).
    Builtin,
    /// Served by an external MCP server.
    #[allow(dead_code)]
    MCP,
    /// Wraps another agent as a callable subagent.
    Subagent,
    /// Registered directly by the user.
    Custom,
}

/// The underlying function implementation for a [`ToolRuntime`].
enum ToolFuncKind {
    /// Synchronous tool — returns [`Value`] directly, no future.
    Sync(Arc<ToolSyncFunc>),
    /// Asynchronous tool — returns [`Value`] via a future.
    Async(Arc<ToolAsyncFunc>),
    /// Streaming tool — yields [`StreamingToolOutput`] items.
    Streaming(Arc<ToolStreamingFunc>),
}

impl Clone for ToolFuncKind {
    fn clone(&self) -> Self {
        match self {
            ToolFuncKind::Sync(f) => ToolFuncKind::Sync(f.clone()),
            ToolFuncKind::Async(f) => ToolFuncKind::Async(f.clone()),
            ToolFuncKind::Streaming(f) => ToolFuncKind::Streaming(f.clone()),
        }
    }
}

#[derive(Clone)]
pub struct ToolRuntime {
    desc: ToolDesc,
    func: ToolFuncKind,
    pub(crate) kind: ToolKind,
}

impl ToolRuntime {
    /// Create a user-defined synchronous tool. `kind` is set to [`ToolKind::Custom`].
    pub fn new_sync(desc: ToolDesc, f: Arc<ToolSyncFunc>) -> Self {
        Self {
            desc,
            func: ToolFuncKind::Sync(f),
            kind: ToolKind::Custom,
        }
    }

    /// Create a user-defined async tool. `kind` is set to [`ToolKind::Custom`].
    pub fn new_async(desc: ToolDesc, f: Arc<ToolAsyncFunc>) -> Self {
        Self {
            desc,
            func: ToolFuncKind::Async(f),
            kind: ToolKind::Custom,
        }
    }

    /// Create a user-defined streaming tool. `kind` is set to [`ToolKind::Custom`].
    pub fn new_streaming(desc: ToolDesc, f: Arc<ToolStreamingFunc>) -> Self {
        Self {
            desc,
            func: ToolFuncKind::Streaming(f),
            kind: ToolKind::Custom,
        }
    }

    #[allow(unused)]
    pub(crate) fn new_sync_with_kind(desc: ToolDesc, f: Arc<ToolSyncFunc>, kind: ToolKind) -> Self {
        Self {
            desc,
            func: ToolFuncKind::Sync(f),
            kind,
        }
    }

    pub(crate) fn new_async_with_kind(
        desc: ToolDesc,
        f: Arc<ToolAsyncFunc>,
        kind: ToolKind,
    ) -> Self {
        Self {
            desc,
            func: ToolFuncKind::Async(f),
            kind,
        }
    }

    pub(crate) fn new_streaming_with_kind(
        desc: ToolDesc,
        f: Arc<ToolStreamingFunc>,
        kind: ToolKind,
    ) -> Self {
        Self {
            desc,
            func: ToolFuncKind::Streaming(f),
            kind,
        }
    }

    pub fn desc(&self) -> &ToolDesc {
        &self.desc
    }

    pub fn can_run(&self, tool_call: &Part) -> anyhow::Result<bool> {
        let (_, name, _) = tool_call
            .as_function()
            .ok_or(anyhow::anyhow!("Part is not function"))?;
        Ok(name == self.desc.name)
    }

    /// Returns `true` if this tool yields incremental [`ToolResultDelta`] items.
    pub fn is_streaming(&self) -> bool {
        matches!(self.func, ToolFuncKind::Streaming(_))
    }

    /// Execute this tool for a sync or async tool call, returning the complete result message.
    ///
    /// Panics if called on a streaming tool — use [`run_stream`](Self::run_stream) instead.
    pub async fn run(&self, tool_call: Part) -> anyhow::Result<Message> {
        let (id, _, args) = tool_call
            .as_function()
            .ok_or(anyhow::anyhow!("Part is not function"))?;
        let result = match &self.func {
            ToolFuncKind::Sync(f) => f(args.clone()),
            ToolFuncKind::Async(f) => f(args.clone()).await,
            ToolFuncKind::Streaming(_) => {
                anyhow::bail!(
                    "run() called on a streaming tool '{}'; use run_stream() instead",
                    self.desc.name
                )
            }
        };
        let mut msg = Message::new(Role::Tool).with_contents([Part::Value { value: result }]);
        if let Some(id) = id {
            msg = msg.with_id(id);
        }
        Ok(msg)
    }

    /// Execute this streaming tool, returning a stream of [`StreamingToolOutput`] items.
    ///
    /// The stream yields zero or more [`StreamingToolOutput::Delta`] items followed by
    /// exactly one [`StreamingToolOutput::Result`] item.
    ///
    /// Panics if called on a non-streaming tool — use [`run`](Self::run) instead.
    pub fn run_stream(
        &self,
        args: Value,
    ) -> Pin<Box<dyn Stream<Item = StreamingToolOutput> + Send>> {
        match &self.func {
            ToolFuncKind::Streaming(f) => {
                use futures::StreamExt as _;
                let fut = f(args);
                Box::pin(futures::stream::once(fut).flatten())
            }
            _ => panic!(
                "run_stream() called on a non-streaming tool '{}'; use run() instead",
                self.desc.name
            ),
        }
    }
}

pub struct ToolSet {
    tools: HashMap<String, ToolRuntime>,
}

impl ToolSet {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: String, value: ToolRuntime) -> Option<ToolRuntime> {
        self.tools.insert(key, value)
    }

    pub fn remove(&mut self, key: &str) -> Option<ToolRuntime> {
        self.tools.remove(key)
    }

    pub fn with_builtin(mut self, provider: &BuiltinToolProvider) -> Self {
        match provider {
            BuiltinToolProvider::WebSearch {} => {
                let tool = web_search::build_web_search_tool();
                self.tools.insert("web_search".to_string(), tool);
                self
            }
        }
    }

    pub fn with_mcp(self, provider: &MCPToolProvider) -> Self {
        match provider {
            MCPToolProvider::Stdio { command: _ } => todo!(),
            MCPToolProvider::StreamableHTTP { url: _ } => todo!(),
        }
    }

    /// Wrap an existing in-process [`AgentRuntime`] as a tool named `name`.
    ///
    /// The tool exposes a single `task` string parameter; when called it forwards
    /// the task to the subagent and yields intermediate assistant messages as
    /// [`ToolResultDelta`] items, with the final assistant response as the tool result.
    pub fn with_subagent_in_memory(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<TokioMutex<AgentRuntime>>,
    ) -> Self {
        let name = name.into();
        let tool = subagent::build_in_memory_subagent_tool(&name, &description.into(), agent);
        self.tools.insert(name, tool);
        self
    }

    /// Discover a remote A2A agent at `url` and register it as a tool.
    ///
    /// Fetches the agent card from `{url}/.well-known/agent-card.json` to obtain
    /// the tool name and description, then builds a tool that forwards calls via
    /// JSON-RPC `message/send`.
    pub async fn with_subagent_a2a(mut self, url: impl Into<String>) -> anyhow::Result<Self> {
        let url = url.into();
        let card = subagent::a2a::discover(&url).await?;
        let name = card.name.clone();
        let tool = subagent::build_a2a_subagent_tool(&card, url);
        self.tools.insert(name, tool);
        Ok(self)
    }

    pub fn get(&self, key: &str) -> Option<&ToolRuntime> {
        self.tools.values().find(|t| t.desc.name == key)
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().into_iter().collect::<Vec<_>>()
    }
}
