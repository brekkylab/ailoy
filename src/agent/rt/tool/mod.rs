use std::{collections::HashMap, sync::Arc};

use futures::future::BoxFuture;
use tokio::sync::Mutex as TokioMutex;

use crate::{
    agent::{AgentRuntime, BuiltinToolProvider, MCPToolProvider},
    datatype::Value,
    message::{Message, Part, Role, ToolDesc},
};

mod convert_pdf_to_md;
mod python_repl;
mod subagent;
mod web_search;

pub type ToolFunc = dyn Fn(Value) -> BoxFuture<'static, Value> + Send + Sync;

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

#[derive(Clone)]
pub struct ToolRuntime {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
    pub(crate) kind: ToolKind,
}

impl ToolRuntime {
    /// Create a user-defined tool. `kind` is set to [`ToolKind::Custom`].
    pub fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self {
            desc,
            f,
            kind: ToolKind::Custom,
        }
    }

    pub(crate) fn new_with_kind(desc: ToolDesc, f: Arc<ToolFunc>, kind: ToolKind) -> Self {
        Self { desc, f, kind }
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

    pub async fn run(&self, tool_call: Part) -> anyhow::Result<Message> {
        let (id, _, args) = tool_call
            .as_function()
            .ok_or(anyhow::anyhow!("Part is not function"))?;
        let result = (self.f)(args.clone()).await;
        let mut msg = Message::new(Role::Tool).with_contents([Part::Value { value: result }]);
        if let Some(id) = id {
            msg = msg.with_id(id);
        }
        Ok(msg)
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

    pub async fn with_builtin(mut self, provider: &BuiltinToolProvider) -> anyhow::Result<Self> {
        match provider {
            BuiltinToolProvider::WebSearch {} => {
                let tool = web_search::build_web_search_tool();
                self.tools.insert("web_search".to_string(), tool);
                Ok(self)
            }
            BuiltinToolProvider::ConvertPdfToMd {} => {
                let tool = convert_pdf_to_md::build_convert_pdf_to_md_tool().await?;
                self.tools.insert("convert_pdf_to_md".to_string(), tool);
                Ok(self)
            }
            BuiltinToolProvider::PythonRepl {
                python_version,
                venv_path,
                packages,
            } => {
                let config = python_repl::PythonReplConfig {
                    python_version: python_version.clone(),
                    venv_path: venv_path.clone(),
                    packages: packages.clone(),
                };
                let tool = python_repl::build_python_repl_tool(config).await?;
                self.tools.insert("python_repl".to_string(), tool);
                Ok(self)
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
    /// The tool exposes a single `task` string parameter; when called it forwards the task to
    /// the agent and returns its first text response.
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
    /// Fetches the agent card from `{url}/.well-known/agent-card.json` to obtain the tool
    /// name and description, then builds a tool that forwards calls via JSON-RPC
    /// `message/send`.
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
