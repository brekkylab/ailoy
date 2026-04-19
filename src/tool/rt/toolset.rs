use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

use crate::{
    agent::{AgentProvider, AgentRuntime},
    message::ToolDesc,
    tool::{BuiltinToolProvider, MCPToolProvider, ToolFunc, ToolProvider, ToolRuntime},
    tool_impl::{
        builtins::{
            PythonReplConfig, build_python_repl_tool, make_web_search_func,
            make_web_search_tool_desc,
        },
        make_a2a_tool, make_subagent_tool,
    },
};

#[derive(Clone)]
pub struct ToolSet {
    tools: HashMap<String, (ToolDesc, Arc<ToolFunc>)>,
}

impl ToolSet {
    /// Build a [`ToolSet`] from the tool providers declared in `provider`.
    ///
    /// Uses a two-pass strategy to avoid infinite recursion:
    ///
    /// 1. **Pass 1** — all non-SubAgent providers are initialised first.
    /// 2. **Pass 2** — each `SubAgent` provider creates an [`AgentRuntime`] that
    ///    receives the tool set produced by pass 1, so sub-agents can call any
    ///    regular tool but cannot themselves contain further sub-agents.
    /// Create an empty [`ToolSet`].
    ///
    /// Useful in tests or when tools are registered manually via [`insert`](Self::insert).
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Build a [`ToolSet`] from the tool providers declared in `provider`.
    ///
    /// Uses a two-pass strategy to avoid infinite recursion:
    ///
    /// 1. **Pass 1** — all non-SubAgent providers are initialised first.
    /// 2. **Pass 2** — each `SubAgent` provider creates an [`AgentRuntime`] that
    ///    receives the tool set produced by pass 1, so sub-agents can call any
    ///    regular tool but cannot themselves contain further sub-agents.
    pub async fn from_providers(provider: &AgentProvider) -> anyhow::Result<Self> {
        let mut this = Self::new();

        // Pass 1: initialise all non-SubAgent tools.
        for tool_provider in &provider.tools {
            match tool_provider {
                ToolProvider::Builtin(builtin_tool_provider) => match builtin_tool_provider {
                    BuiltinToolProvider::WebSearch {} => {
                        this.tools.insert(
                            "web_search".into(),
                            (
                                make_web_search_tool_desc(),
                                Arc::new(make_web_search_func()),
                            ),
                        );
                    }
                    BuiltinToolProvider::PythonRepl {
                        python_version,
                        venv_path,
                        packages,
                    } => {
                        let tool_runtime = build_python_repl_tool(PythonReplConfig {
                            python_version: python_version.clone(),
                            venv_path: venv_path.clone(),
                            packages: packages.clone(),
                        })
                        .await?;
                        this.tools.insert(
                            "python_repl".into(),
                            (tool_runtime.get_desc().clone(), tool_runtime.get_func()),
                        );
                    }
                },
                // Skip sub-agents in the first pass.
                ToolProvider::SubAgent { .. } => {}
                ToolProvider::A2A { url } => {
                    let tool_runtime = make_a2a_tool(url.clone()).await?;
                    this.tools.insert(
                        tool_runtime.get_desc().name.clone(),
                        (tool_runtime.get_desc().clone(), tool_runtime.get_func()),
                    );
                }
                ToolProvider::MCP(mcptool_provider) => match mcptool_provider {
                    MCPToolProvider::Stdio { command: _ } => todo!(),
                    MCPToolProvider::StreamableHTTP { url: _ } => todo!(),
                },
            }
        }

        // Pass 2: initialise SubAgent tools using the toolset built above.
        for tool_provider in &provider.tools {
            if let ToolProvider::SubAgent { spec } = tool_provider {
                let card = spec.card.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "SubAgent spec must have a card (name + description) to be \
                         registered as a tool"
                    )
                })?;
                let tool_name = card.name.clone();
                let sub_agent =
                    AgentRuntime::try_from_toolset(spec.clone(), provider.clone(), &this)?;
                let sub_agent = Arc::new(Mutex::new(sub_agent));
                let tool_runtime = make_subagent_tool(card, sub_agent);
                this.tools.insert(
                    tool_name,
                    (tool_runtime.get_desc().clone(), tool_runtime.get_func()),
                );
            }
        }

        Ok(this)
    }

    /// Insert a tool by key.
    ///
    /// `f` can be a [`ToolFunc`] (wrapped in an `Arc` automatically) or an
    /// existing `Arc<ToolFunc>` — e.g. one obtained from [`ToolRuntime::get_func`].
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        desc: ToolDesc,
        f: impl Into<Arc<ToolFunc>>,
    ) -> Option<(ToolDesc, Arc<ToolFunc>)> {
        self.tools.insert(key.into(), (desc, f.into()))
    }

    pub fn remove(&mut self, key: impl AsRef<str>) -> Option<(ToolDesc, Arc<ToolFunc>)> {
        self.tools.remove(key.as_ref())
    }

    pub fn make_runtime(&self, key: impl AsRef<str>) -> Option<ToolRuntime> {
        self.tools
            .get(key.as_ref())
            .map(|(desc, f)| ToolRuntime::new(desc.clone(), Arc::clone(f)))
    }
}
