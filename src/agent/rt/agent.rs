use crate::{
    agent::{AgentProvider, AgentSpec, LangModelRuntime, ToolProvider, ToolRuntime, ToolSet},
    message::{Message, Part, Role},
};

pub struct AgentRuntime<'a> {
    lm: LangModelRuntime,
    tools: Vec<ToolRuntime<'a>>,
    history: Vec<Message>,
}

impl<'a> AgentRuntime<'a> {
    pub fn new(spec: AgentSpec, provider: AgentProvider) -> Self {
        // Prepare toolsets
        let tool_set = provider
            .tools
            .iter()
            .fold(ToolSet::new(), |ts, tool_provider| match tool_provider {
                ToolProvider::Builtin { name } => ts.with_builtin(name),
                ToolProvider::MCP(mcp) => ts.with_mcp(mcp),
            });

        // Collect tools from toolsets
        let tools = spec
            .tools
            .iter()
            .filter_map(|name| tool_set.get(name).cloned())
            .collect();

        // Initialize histroy
        let history = spec
            .instruction
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        // Create `AgentRuntime`
        Self {
            lm: LangModelRuntime::new(spec.lm, provider.lm),
            tools,
            history,
        }
    }

    pub async fn run(&mut self, query: Message) -> anyhow::Result<Message> {
        self.history.push(query);

        let tool_descs: Vec<_> = self.tools.iter().map(|t| t.desc().clone()).collect();

        loop {
            let output = self.lm.run(&self.history, &tool_descs).await?;
            let assistant_msg = output.message;
            self.history.push(assistant_msg.clone());

            let tool_calls = assistant_msg.tool_calls.unwrap_or_default();
            if tool_calls.is_empty() {
                break;
            }

            for tool_call in tool_calls {
                let tool = self
                    .tools
                    .iter()
                    .find(|t| t.can_run(&tool_call).unwrap_or(false))
                    .ok_or_else(|| anyhow::anyhow!("No tool found for call"))?;
                let tool_msg = tool.run(tool_call).await?;
                self.history.push(tool_msg);
            }
        }

        self.history
            .iter()
            .rfind(|m| m.role == Role::Assistant)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No assistant response"))
    }
}
