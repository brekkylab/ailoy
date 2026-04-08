use std::pin::Pin;

use futures::{Stream, StreamExt as _};

use crate::{
    agent::{AgentProvider, AgentSpec, LangModelRuntime, ToolProvider, ToolRuntime, ToolSet},
    message::{FinishReason, Message, MessageOutput, Part, Role},
};

pub struct AgentRuntime {
    lm: LangModelRuntime,
    tools: Vec<ToolRuntime>,
    history: Vec<Message>,
}

impl AgentRuntime {
    pub async fn new(
        spec: AgentSpec,
        provider: AgentProvider,
        tool_set: ToolSet,
    ) -> anyhow::Result<Self> {
        // Prepare toolsets
        let mut tool_set = tool_set;
        for tool_provider in &provider.tools {
            tool_set = match tool_provider {
                ToolProvider::Builtin(builtin) => tool_set.with_builtin(builtin).await?,
                ToolProvider::MCP(mcp) => tool_set.with_mcp(mcp),
            };
        }

        // Collect tools from toolsets
        let tools = spec
            .tools
            .iter()
            .filter_map(|name| tool_set.get(name).cloned())
            .collect();

        // Initialize history
        let history = spec
            .instruction
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        // Create `AgentRuntime`
        Ok(Self {
            lm: LangModelRuntime::new(spec.lm, provider.lm),
            tools,
            history,
        })
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

    pub fn get_history(&self) -> Vec<Message> {
        self.history.clone()
    }

    pub fn set_history(&mut self, history: Vec<Message>) {
        self.history = history;
    }

    pub fn stream_turn(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageOutput>> + '_>> {
        Box::pin(async_stream::stream! {
            self.history.push(query);
            let tool_descs: Vec<_> = self.tools.iter().map(|t| t.desc().clone()).collect();

            loop {
                let output = match self.lm.run(&self.history, &tool_descs, &Default::default()).await {
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

    use super::*;
    use crate::{
        agent::{BuiltinToolProvider, LangModelProvider, ToolFunc},
        message::{Part, Role, ToolDescBuilder},
        to_value,
    };

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    #[tokio::test]
    #[ignore = "requires network + OPENAI_API_KEY"]
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

        let mut agent = AgentRuntime::new(
            AgentSpec::new("gpt-4").with_tools(["temperature"]),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![],
            },
            tool_set,
        )
        .await
        .unwrap();

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

    /// End-to-end test: agent uses python_repl to generate a numpy sine-wave
    /// chart with matplotlib, saves it as a PNG, and we validate the file.
    ///
    /// Requires: OPENAI_API_KEY env var, network access.
    #[tokio::test]
    #[ignore = "requires network + OPENAI_API_KEY"]
    async fn test_python_repl_numpy_matplotlib_chart() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        // Temp dir for the chart output — kept alive for the duration of the test.
        let chart_dir = tempfile::tempdir().expect("failed to create temp dir");
        let chart_path = chart_dir.path().join("sine_chart.png");

        let prompt = format!(
            "Install numpy and matplotlib, then write Python code that:\n\
             1. Generates 200 points of a sine wave using numpy (x: 0 to 4π)\n\
             2. Plots the sine wave with matplotlib\n\
             3. Saves the figure to '{}'\n\
             \n\
             Use pip_install for the packages. Do not call plt.show().",
            chart_path.display()
        );

        let mut agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini").with_tools(["python_repl"]),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![ToolProvider::Builtin(BuiltinToolProvider::PythonRepl {
                    python_version: None,
                    venv_path: None,
                    packages: vec![],
                })],
            },
            ToolSet::new(),
        )
        .await
        .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(prompt)]);
        let mut outputs = vec![];
        let mut strm = agent.stream_turn(query);
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            println!("{}", out);
            outputs.push(out);
        }

        // ── assertions ────────────────────────────────────────────────────────

        // 1. The agent must have made at least one python_repl tool call.
        let tool_outputs: Vec<_> = outputs
            .iter()
            .filter(|o| o.message.role == Role::Tool)
            .collect();
        assert!(
            !tool_outputs.is_empty(),
            "expected at least one python_repl tool call, got none"
        );

        // 2. Every tool call that returned must have exit_code 0 (success).
        for tool_out in &tool_outputs {
            for part in &tool_out.message.contents {
                if let Some(v) = part.as_value() {
                    let exit_code = v
                        .pointer("/exit_code")
                        .and_then(|c| c.as_integer())
                        .unwrap_or(-1);
                    let stdout = v.pointer("/stdout").and_then(|s| s.as_str()).unwrap_or("");
                    let stderr = v.pointer("/stderr").and_then(|s| s.as_str()).unwrap_or("");
                    assert_eq!(
                        exit_code, 0,
                        "python_repl returned non-zero exit code.\nstdout: {stdout}\nstderr: {stderr}"
                    );
                }
            }
        }

        // 3. Chart file must exist.
        assert!(
            chart_path.exists(),
            "chart file was not created at {:?}",
            chart_path
        );

        // 4. File must be a valid PNG (check 8-byte magic signature).
        const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
        let header = std::fs::read(&chart_path).unwrap();
        assert!(
            header.starts_with(PNG_MAGIC),
            "file at {:?} is not a valid PNG (got {:?})",
            chart_path,
            &header[..header.len().min(8)]
        );

        // 5. The final message must be an assistant text reply.
        let final_msg = outputs.last().expect("expected at least one output");
        assert_eq!(final_msg.message.role, Role::Assistant);
        assert!(
            final_msg.message.contents.iter().any(|p| p.is_text()),
            "expected assistant to summarise its work in text"
        );
    }
}
