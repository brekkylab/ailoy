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
    pub fn new(spec: AgentSpec, provider: AgentProvider, tool_set: ToolSet<'a>) -> Self {
        // Prepare toolsets
        let tool_set =
            provider
                .tools
                .iter()
                .fold(tool_set, |ts, tool_provider| match tool_provider {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use url::Url;

    use crate::{
        agent::{LangModelAPISchema, LangModelProvider, ToolFunc},
        message::{Part, Role, ToolDescBuilder},
        to_value,
    };

    use super::*;

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    #[tokio::test]
    async fn test_run_agent() {
        dotenvy::dotenv().ok();

        let temperature_desc = ToolDescBuilder::new("temperature")
            .description("Get the current temperature for a given city")
            .parameters(to_value!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }))
            .build();

        let temperature_fn: Arc<ToolFunc> = Arc::new(|_args| {
            Box::pin(async move { to_value!(25) }) as futures::future::BoxFuture<'static, _>
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature".to_string(),
            ToolRuntime::new(temperature_desc, temperature_fn),
        );

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");
        let url = Url::parse("https://api.openai.com/v1/chat/completions").unwrap();

        let mut agent = AgentRuntime::new(
            AgentSpec::new("gpt-4").with_tools(["temperature"]),
            AgentProvider {
                lm: LangModelProvider::API {
                    schema: LangModelAPISchema::ChatCompletion,
                    url,
                    api_key: Some(api_key),
                },
                tools: vec![],
            },
            tool_set,
        );

        let query = Message::new(Role::User)
            .with_contents([Part::text("What is the current temperature in Seoul?")]);

        let resp = agent.run(query).await.unwrap();

        println!("{}", resp);

        // The agent should have invoked the tool and returned a final text answer
        assert_eq!(resp.role, Role::Assistant);
        assert!(
            resp.contents.iter().any(|p| p.is_text()),
            "Expected final assistant message to contain text"
        );
    }
}
