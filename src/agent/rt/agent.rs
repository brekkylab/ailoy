use std::pin::Pin;

use futures::{Stream, StreamExt as _};

use crate::{
    agent::{AgentProvider, AgentSpec, LangModelRuntime, ToolProvider, ToolRuntime, ToolSet},
    message::{FinishReason, Message, MessageOutput, Part, Role},
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

    pub fn run(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Future<Output = anyhow::Result<Message>> + '_>> {
        Box::pin(async move {
            let mut last = None;
            let mut strm = self.stream_turn(query);
            while let Some(output) = strm.next().await {
                last = Some(output?);
            }
            last.map(|o: MessageOutput| o.message)
                .ok_or_else(|| anyhow::anyhow!("No assistant response"))
        })
    }

    pub fn stream_turn(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageOutput>> + '_>> {
        Box::pin(async_stream::stream! {
            self.history.push(query);
            let tool_descs: Vec<_> = self.tools.iter().map(|t| t.desc().clone()).collect();

            loop {
                let output = match self.lm.run(&self.history, &tool_descs).await {
                    Ok(o) => o,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                let assistant_msg = output.message.clone();
                self.history.push(assistant_msg.clone());
                yield Ok(output);

                let tool_calls = assistant_msg.tool_calls.unwrap_or_default();
                if tool_calls.is_empty() {
                    break;
                }
                for tool_call in tool_calls {
                    let tool = match self
                        .tools
                        .iter()
                        .find(|t| t.can_run(&tool_call).unwrap_or(false))
                    {
                        Some(t) => t.clone(),
                        None => {
                            yield Err(anyhow::anyhow!("No tool found for call"));
                            return;
                        }
                    };
                    let tool_msg = match tool.run(tool_call).await {
                        Ok(m) => m,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };
                    self.history.push(tool_msg.clone());
                    yield Ok(MessageOutput {
                        message: tool_msg,
                        finish_reason: FinishReason::Stop {},
                    });
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt as _;
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

        let mut strm = agent.stream_turn(query);
        let mut outputs = vec![];
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("{}", output);
            outputs.push(output);
        }

        let resp = outputs.last().expect("Expected at least one output");

        // The agent should have invoked the tool and returned a final text answer
        assert_eq!(resp.message.role, Role::Assistant);
        assert!(
            resp.message.contents.iter().any(|p| p.is_text()),
            "Expected final assistant message to contain text"
        );
    }
}
