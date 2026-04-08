use std::{collections::HashMap, sync::Arc};

use futures::future::BoxFuture;

use crate::{
    agent::{BuiltinToolProvider, MCPToolProvider},
    datatype::Value,
    message::{Message, Part, Role, ToolDesc},
};

mod python_repl;
mod web_search;

pub type ToolFunc = dyn Fn(Value) -> BoxFuture<'static, Value> + Send + Sync;

#[derive(Clone)]
pub struct ToolRuntime {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
}

impl ToolRuntime {
    pub fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self { desc, f }
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

    pub fn get(&self, key: &str) -> Option<&ToolRuntime> {
        self.tools.values().find(|t| t.desc.name == key)
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().into_iter().collect::<Vec<_>>()
    }
}
