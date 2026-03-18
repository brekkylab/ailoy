use std::{collections::HashMap, sync::Arc};

use futures::future::BoxFuture;

use crate::{
    agent::MCPToolProvider,
    datatype::Value,
    message::{Message, Part, Role, ToolDesc},
};

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

    pub fn with_builtin(self, name: &str) -> Self {
        if name == "knowledge" {
            todo!()
        } else {
            // @jhlee: Should we raise?
            self
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
}
