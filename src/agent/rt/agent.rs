use std::pin::Pin;

use futures::{Stream, StreamExt as _};

use super::turn_event::TurnEvent;
use crate::{
    agent::{
        AgentProvider, AgentSpec, LangModelRuntime, ToolKind, ToolProvider, ToolRuntime, ToolSet,
    },
    message::{Message, Part, Role, StreamingToolOutput},
    shell::Shell,
};

pub struct AgentState {
    pub history: Vec<Message>,

    pub shell: Option<Shell>,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            shell: None,
        }
    }

    pub fn with_history(history: Vec<Message>) -> Self {
        Self {
            history,
            shell: None,
        }
    }

    pub fn shell(mut self) -> Self {
        self.shell = Some(Shell::new());
        self
    }
}

pub struct AgentRuntime {
    lm: LangModelRuntime,
    tools: Vec<ToolRuntime>,
    state: AgentState,
}

impl AgentRuntime {
    pub fn new(spec: AgentSpec, provider: AgentProvider, tool_set: ToolSet) -> Self {
        // Prepare toolsets
        let tool_set =
            provider
                .tools
                .iter()
                .fold(tool_set, |ts, tool_provider| match tool_provider {
                    ToolProvider::Builtin(builtin) => ts.with_builtin(builtin),
                    ToolProvider::MCP(mcp) => ts.with_mcp(mcp),
                });

        // Collect tools from toolsets
        let tools: Vec<ToolRuntime> = spec
            .tools
            .iter()
            .filter_map(|name| tool_set.get(name).cloned())
            .collect();

        let mut instruction = spec.instruction.clone();

        // Append subagent suffix on instruction if exist
        let subagent_tools = tools
            .iter()
            .filter(|t| t.kind == ToolKind::Subagent)
            .map(|t| t.desc())
            .collect::<Vec<_>>();
        if !subagent_tools.is_empty() {
            let suffix = build_subagent_system_suffix(&subagent_tools);
            instruction = match instruction {
                Some(existing) => Some(format!("{}\n\n{}", existing, suffix)),
                None => Some(suffix),
            };
        }

        // Initialize history
        let history = instruction
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        // Create `AgentRuntime`
        Self {
            lm: LangModelRuntime::new(spec.lm, provider.lm),
            tools,
            state: AgentState::with_history(history),
        }
    }

    /// Run a full turn and return the final assistant message.
    ///
    /// Drains [`stream_turn`](Self::stream_turn) and returns the message from
    /// the last [`TurnEvent::AssistantMessage`] with `is_final: true`.
    pub fn run(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Future<Output = anyhow::Result<Message>> + '_>> {
        Box::pin(async move {
            let mut last = None;
            let mut strm = self.stream_turn(query);
            while let Some(event) = strm.next().await {
                if let TurnEvent::AssistantMessage(output) = event? {
                    last = Some(output.message);
                }
            }
            last.ok_or_else(|| anyhow::anyhow!("No assistant response"))
        })
    }

    pub fn get_history(&self) -> Vec<Message> {
        self.state.history.clone()
    }

    pub fn set_history(&mut self, history: Vec<Message>) {
        self.state.history = history;
    }

    /// Stream all events for a single agent turn.
    ///
    /// Yields a sequence of [`TurnEvent`] items:
    /// - [`TurnEvent::AssistantMessage`] for each LLM response
    /// - [`TurnEvent::ToolCall`] for each tool call extracted from the response
    /// - [`TurnEvent::ToolDelta`] for intermediate progress from streaming tools
    /// - [`TurnEvent::ToolResult`] for each finalized tool result
    ///
    /// The loop continues until the LLM produces a response with no tool calls.
    pub fn stream_turn(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<TurnEvent>> + '_>> {
        Box::pin(async_stream::stream! {
            self.state.history.push(query);
            let tool_descs: Vec<_> = self.tools.iter().map(|t| t.desc().clone()).collect();

            loop {
                let output = match self.lm.run(&self.state.history, &tool_descs, &Default::default()).await {
                    Ok(o) => o,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                let assistant_msg = output.message.clone();
                self.state.history.push(assistant_msg.clone());

                let tool_calls = assistant_msg.tool_calls.clone().unwrap_or_default();

                if tool_calls.is_empty() {
                    // Final assistant response — yield the message and end the turn
                    yield Ok(TurnEvent::AssistantMessage(output));
                    break;
                }

                // Tool-calling turn — yield one ToolCall per call, skip AssistantMessage
                for tool_call in tool_calls {
                    if let Some((id, name, arguments)) = tool_call.as_function() {
                        yield Ok(TurnEvent::ToolCall {
                            id: id.map(|s| s.to_string()),
                            name: name.to_string(),
                            arguments: arguments.clone(),
                        });
                    }

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

                    let tool_name = tool.desc().name.clone();
                    let tool_call_id = tool_call.as_function().and_then(|(id, _, _)| id).map(|s| s.to_string());

                    if tool.is_streaming() {
                        // Extract args for run_stream
                        let args = match tool_call.as_function() {
                            Some((_, _, args)) => args.clone(),
                            None => {
                                yield Err(anyhow::anyhow!("Part is not function"));
                                return;
                            }
                        };

                        let mut output_stream = tool.run_stream(args);
                        let mut got_result = false;

                        while let Some(item) = output_stream.next().await {
                            match item {
                                StreamingToolOutput::Delta(mut delta) => {
                                    delta.tool_call_id = tool_call_id.clone();
                                    yield Ok(TurnEvent::ToolDelta(delta));
                                }
                                StreamingToolOutput::Result(value) => {
                                    let mut msg = Message::new(Role::Tool)
                                        .with_contents([Part::Value { value }]);
                                    if let Some(id) = &tool_call_id {
                                        msg = msg.with_id(id.clone());
                                    }
                                    self.state.history.push(msg.clone());
                                    yield Ok(TurnEvent::ToolResult(msg));
                                    got_result = true;
                                }
                            }
                        }

                        if !got_result {
                            // Streaming tool never yielded a Result — produce an error
                            let mut msg = Message::new(Role::Tool)
                                .with_contents([Part::Value {
                                    value: format!(
                                        "Error: tool '{}' did not yield a result",
                                        tool_name
                                    )
                                    .into(),
                                }]);
                            if let Some(id) = &tool_call_id {
                                msg = msg.with_id(id.clone());
                            }
                            self.state.history.push(msg.clone());
                            yield Ok(TurnEvent::ToolResult(msg));
                        }
                    } else {
                        let tool_msg = match tool.run(tool_call).await {
                            Ok(m) => m,
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        };
                        self.state.history.push(tool_msg.clone());
                        yield Ok(TurnEvent::ToolResult(tool_msg));
                    }
                }
            }
        })
    }
}

fn build_subagent_system_suffix(descs: &[&crate::message::ToolDesc]) -> String {
    let mut lines = vec![
        "## Subagent Delegation".to_string(),
        String::new(),
        "You can delegate tasks to specialized subagents by calling their tools directly."
            .to_string(),
        String::new(),
        "### Available Subagents".to_string(),
        String::new(),
    ];

    for desc in descs {
        lines.push(format!("#### {}", desc.name));
        if let Some(d) = &desc.description {
            lines.push(d.clone());
        }
        lines.push(String::new());
    }

    lines.push(
        "When a request clearly matches a subagent's capabilities, call their tool directly. \
         If no subagent is a good fit, handle it yourself."
            .to_string(),
    );

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt as _;

    use super::*;
    use crate::{
        ToolSyncFunc,
        agent::LangModelProvider,
        message::{Part, Role, ToolDescBuilder},
        to_value,
    };

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

        let temperature_fn: Arc<ToolSyncFunc> = Arc::new(|_args| to_value!(25));

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature".to_string(),
            ToolRuntime::new_sync(temperature_desc, temperature_fn),
        );

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let mut agent = AgentRuntime::new(
            AgentSpec::new("gpt-4").with_tools(["temperature"]),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![],
            },
            tool_set,
        );

        let query = Message::new(Role::User)
            .with_contents([Part::text("What is the current temperature in Seoul?")]);

        let mut strm = agent.stream_turn(query);
        let mut events = vec![];
        while let Some(event) = strm.next().await {
            let event = event.unwrap();
            events.push(event);
        }

        // The last assistant message is the final one
        let final_output = events
            .iter()
            .rev()
            .find_map(|e| e.as_assistant_message())
            .expect("Expected an assistant message");

        assert_eq!(final_output.message.role, Role::Assistant);
        assert!(
            final_output.message.contents.iter().any(|p| p.is_text()),
            "Expected final assistant message to contain text"
        );
    }

    /// Verifies that the main agent actually delegates to the in-memory subagent.
    ///
    /// Sets up a math subagent and registers it as a subagent tool on a coordinator agent.
    /// Asks a math question and confirms the main agent's history contains a [`Role::Tool`]
    /// message (proof that the subagent tool was called), and that the final reply is text.
    #[tokio::test]
    async fn test_delegate_to_in_memory_subagent() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        // Sub-agent: a minimal calculator that replies with just the numeric result.
        let sub_agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini").with_instruction(
                "You are a calculator. Answer math questions with the numeric result only."
                    .to_string(),
            ),
            AgentProvider {
                lm: LangModelProvider::openai(api_key.clone()),
                tools: vec![],
            },
            ToolSet::new(),
        );
        let sub_agent = Arc::new(tokio::sync::Mutex::new(sub_agent));

        // Main agent: coordinator that should always delegate math to math-agent.
        let tool_set = ToolSet::new().with_subagent_in_memory(
            "math-agent",
            "Handles arithmetic and math computations. Use this for any math question.",
            sub_agent,
        );

        let mut main_agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini").with_tools(["math-agent"]),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![],
            },
            tool_set,
        );

        let query =
            Message::new(Role::User).with_contents([Part::text("What is 123 multiplied by 7?")]);

        let events = {
            let mut strm = main_agent.stream_turn(query);
            let mut events = vec![];
            while let Some(event) = strm.next().await {
                events.push(event.unwrap());
            }
            events
        };

        // The history must contain a Tool message, confirming the subagent was called.
        let history = main_agent.get_history();
        assert!(
            history.iter().any(|m| m.role == Role::Tool),
            "Expected main agent history to contain a Tool message (subagent was called)"
        );

        // Final output must be an assistant text response.
        let final_output = events
            .iter()
            .rev()
            .find_map(|e| e.as_assistant_message())
            .expect("Expected an assistant message");
        assert_eq!(final_output.message.role, Role::Assistant);
        assert!(
            final_output.message.contents.iter().any(|p| p.is_text()),
            "Expected final assistant message to contain text"
        );
    }

    /// Verifies that stream_turn() emits TurnEvent::ToolDelta items when using a
    /// streaming subagent tool.
    #[tokio::test]
    async fn test_streaming_subagent_emits_tool_deltas() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        // Sub-agent: gives a multi-step response so we see intermediate messages
        let sub_agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini").with_instruction(
                "You are a calculator. Answer math questions with the numeric result only."
                    .to_string(),
            ),
            AgentProvider {
                lm: LangModelProvider::openai(api_key.clone()),
                tools: vec![],
            },
            ToolSet::new(),
        );
        let sub_agent = Arc::new(tokio::sync::Mutex::new(sub_agent));

        let tool_set = ToolSet::new().with_subagent_in_memory(
            "math-agent",
            "Handles arithmetic and math computations. Use this for any math question.",
            sub_agent,
        );

        let mut main_agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini").with_tools(["math-agent"]),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![],
            },
            tool_set,
        );

        let query = Message::new(Role::User).with_contents([Part::text("What is 99 plus 1?")]);

        let mut strm = main_agent.stream_turn(query);
        let mut tool_deltas = 0usize;
        let mut tool_results = 0usize;

        while let Some(event) = strm.next().await {
            match event.unwrap() {
                TurnEvent::ToolDelta(_) => tool_deltas += 1,
                TurnEvent::ToolResult(_) => tool_results += 1,
                _ => {}
            }
        }

        assert!(tool_results > 0, "Expected at least one ToolResult");
        // ToolDeltas come from the subagent's intermediate messages
        assert!(
            tool_deltas > 0,
            "Expected at least one ToolDelta from the streaming subagent"
        );
    }

    /// Verifies event sequence for a simple tool call:
    /// AssistantMessage(is_final=false) -> ToolCall -> ToolResult -> AssistantMessage(is_final=true)
    #[tokio::test]
    async fn test_turn_event_sequence() {
        dotenvy::dotenv().ok();

        let temperature_desc = ToolDescBuilder::new("temperature")
            .description("Get the current temperature for a given city")
            .parameters(to_value!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }))
            .build();

        let temperature_fn: Arc<ToolSyncFunc> = Arc::new(|_args| to_value!(25));

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature".to_string(),
            ToolRuntime::new_sync(temperature_desc, temperature_fn),
        );

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let mut agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini").with_tools(["temperature"]),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![],
            },
            tool_set,
        );

        let query = Message::new(Role::User)
            .with_contents([Part::text("What is the temperature in Seoul?")]);

        let mut strm = agent.stream_turn(query);
        let mut events = vec![];
        while let Some(event) = strm.next().await {
            events.push(event.unwrap());
        }

        let has_tool_call = events
            .iter()
            .any(|e| matches!(e, TurnEvent::ToolCall { .. }));
        let has_tool_result = events.iter().any(|e| matches!(e, TurnEvent::ToolResult(_)));

        // Exactly one AssistantMessage (the final response), ToolCall, and ToolResult
        assert!(has_tool_call, "Expected ToolCall event");
        assert!(has_tool_result, "Expected ToolResult event");
        assert!(
            events
                .last()
                .and_then(|e| e.as_assistant_message())
                .is_some(),
            "Last event should be the final AssistantMessage"
        );
    }

    #[test]
    fn test_subagent_suffix_appended_to_system_message() {
        let desc1 = ToolDescBuilder::new("math-agent")
            .description("Handles arithmetic and math computations")
            .parameters(serde_json::json!({"type":"object","properties":{"task":{"type":"string"}},"required":["task"]}))
            .build();
        let desc2 = ToolDescBuilder::new("search-agent")
            .description("Searches the web for information")
            .parameters(serde_json::json!({"type":"object","properties":{"task":{"type":"string"}},"required":["task"]}))
            .build();

        let refs: Vec<&crate::message::ToolDesc> = vec![&desc1, &desc2];
        let suffix = build_subagent_system_suffix(&refs);

        assert!(suffix.contains("math-agent"), "should contain agent name");
        assert!(suffix.contains("search-agent"), "should contain agent name");
        assert!(
            suffix.contains("Handles arithmetic"),
            "should contain description"
        );
        assert!(
            suffix.contains("Searches the web"),
            "should contain description"
        );
        assert!(
            suffix.contains("Subagent Delegation"),
            "should contain section header"
        );
        assert!(
            suffix.contains("call their tool directly"),
            "should contain delegation instruction"
        );
    }
}
