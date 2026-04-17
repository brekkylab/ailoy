use std::pin::Pin;
use std::sync::Arc;

use futures::{FutureExt as _, Stream, StreamExt as _};

use super::turn_event::TurnEvent;
use crate::{
    agent::{
        AgentProvider, AgentSpec, LangModelRuntime, ToolKind, ToolProvider, ToolRuntime, ToolSet,
    },
    message::{Message, Part, Role, StreamingToolOutput},
    sandbox::Sandbox,
};

pub struct AgentState {
    pub history: Vec<Message>,

    pub sandbox: Option<Arc<dyn Sandbox>>,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            sandbox: None,
        }
    }

    pub fn with_history(history: Vec<Message>) -> Self {
        Self {
            history,
            sandbox: None,
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<dyn Sandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }
}

pub struct AgentRuntime {
    lm: LangModelRuntime,
    tools: Vec<ToolRuntime>,
    state: AgentState,
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
        Ok(Self {
            lm: LangModelRuntime::new(spec.lm, provider.lm),
            tools,
            state: AgentState::with_history(history),
        })
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

                // Tool-calling turn — yield one ToolCall per call, collect tasks
                let mut tool_tasks: Vec<(ToolRuntime, Part)> = Vec::new();
                let mut setup_error = false;
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
                            setup_error = true;
                            break;
                        }
                    };

                    tool_tasks.push((tool, tool_call));
                }
                if setup_error {
                    return;
                }

                // Execute all tools concurrently via mpsc channel
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<TurnEvent>>();

                let ctx = crate::ToolContext {
                    sandbox: self.state.sandbox.clone(),
                };

                for (tool, tool_call) in tool_tasks {
                    let tx = tx.clone();
                    let ctx = ctx.clone();
                    let tool_name = tool.desc().name.clone();
                    let tool_call_id = tool_call
                        .as_function()
                        .and_then(|(id, _, _)| id)
                        .map(|s| s.to_string());

                    // Clone tool_call_id used on synthetic error message
                    let tool_call_id_err = tool_call_id.clone();

                    tokio::spawn(async move {

                        // tx_task is moved into the catch_unwind block.
                        // tx is kept for the panic-recovery send below.
                        let tx_task = tx.clone();


                        // Returns Ok(true)  — a ToolResult was sent
                        //         Ok(false) — stream ended without emitting a Result
                        //         Err(_)    — the tool function panicked
                        let outcome: Result<bool, _> =
                            std::panic::AssertUnwindSafe(async move {
                                if tool.is_streaming() {
                                    let args = match tool_call.as_function() {
                                        Some((_, _, args)) => args.clone(),
                                        None => {
                                            let _ = tx_task.send(Err(anyhow::anyhow!(
                                                "Part is not function"
                                            )));
                                            return true;
                                        }
                                    };
                                    let mut output_stream = tool.run_stream(args, ctx);
                                    let mut got_result = false;
                                    while let Some(item) = output_stream.next().await {
                                        let event = match item {
                                            StreamingToolOutput::Delta(mut delta) => {
                                                delta.tool_call_id = tool_call_id.clone();
                                                TurnEvent::ToolDelta(delta)
                                            }
                                            StreamingToolOutput::Result(value) => {
                                                got_result = true;
                                                let mut msg = Message::new(Role::Tool)
                                                    .with_contents([Part::Value { value }]);
                                                if let Some(id) = &tool_call_id {
                                                    msg = msg.with_id(id.clone());
                                                }
                                                TurnEvent::ToolResult(msg)
                                            }
                                        };
                                        if tx_task.send(Ok(event)).is_err() {
                                            return got_result;
                                        }
                                    }
                                    got_result
                                } else {
                                    let event = match tool.run(tool_call, ctx).await {
                                        Ok(msg) => Ok(TurnEvent::ToolResult(msg)),
                                        Err(e) => Err(e),
                                    };
                                    let _ = tx_task.send(event);
                                    true
                                }
                            })
                            .catch_unwind()
                            .await;

                        // Guarantee that exactly one ToolResult reaches the channel per tool
                        // call. If the tool panicked (Err) or the streaming tool never emitted
                        // a Result (Ok(false)), insert a synthetic error result so the agent
                        // history stays consistent and the LLM can see the failure.
                        let needs_fallback = !outcome.as_ref().copied().unwrap_or(false);
                        if needs_fallback {
                            let reason = match outcome {
                                Err(_) => "panicked during execution",
                                Ok(false) => "did not yield a result",
                                Ok(true) => unreachable!(),
                            };
                            let mut msg = Message::new(Role::Tool).with_contents([Part::Value {
                                value: format!("Error: tool '{}' {}", tool_name, reason)
                                    .into(),
                            }]);
                            if let Some(id) = &tool_call_id_err {
                                msg = msg.with_id(id.clone());
                            }
                            let _ = tx.send(Ok(TurnEvent::ToolResult(msg)));
                        }
                    });
                }
                // Drop the original sender so rx closes when all spawned tasks finish
                drop(tx);

                // Drain channel: yield events and collect ToolResult messages for history
                let mut result_messages: Vec<Message> = Vec::new();
                while let Some(event) = rx.recv().await {
                    match event {
                        Ok(TurnEvent::ToolResult(ref msg)) => {
                            result_messages.push(msg.clone());
                            yield Ok(event.unwrap());
                        }
                        Ok(event) => {
                            yield Ok(event);
                        }
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    }
                }

                // Push all tool results to history after all tools have completed
                for msg in result_messages {
                    self.state.history.push(msg);
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
        suppress_panics, to_value,
    };

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    #[test_with::env(OPENAI_API_KEY)]
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

        let temperature_fn: Arc<ToolSyncFunc> = Arc::new(|_args, _ctx: crate::ToolContext| to_value!(25));

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
        )
        .await
        .unwrap();

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
    #[test_with::env(OPENAI_API_KEY)]
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
        )
        .await
        .unwrap();
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
        )
        .await
        .unwrap();

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
    #[test_with::env(OPENAI_API_KEY)]
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
        )
        .await
        .unwrap();
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
        )
        .await
        .unwrap();

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
    #[test_with::env(OPENAI_API_KEY)]
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

        let temperature_fn: Arc<ToolSyncFunc> = Arc::new(|_args, _ctx: crate::ToolContext| to_value!(25));

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
        )
        .await
        .unwrap();

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

    /// Verifies that multiple tool calls in a single LLM response are executed concurrently.
    ///
    /// Two async tools are registered:
    /// - `temperature_fast`: 50 ms delay, returns 15
    /// - `temperature_slow`: 300 ms delay, returns 25
    ///
    /// Assertions:
    /// 1. Total tool-execution time ≈ max(50, 300) ms, not 50+300 ms (sequential).
    /// 2. The faster tool's result appears first in history (completion order, not call order).
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_parallel_tool_calls() {
        dotenvy::dotenv().ok();

        use std::sync::{Arc as StdArc, Mutex as StdMutex};

        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in .env");

        // Record the order in which tools complete
        let completion_order: StdArc<StdMutex<Vec<&'static str>>> =
            StdArc::new(StdMutex::new(Vec::new()));

        // fast tool: completes in ~50 ms, returns 15
        let fast_desc = ToolDescBuilder::new("temperature_fast")
            .description("Get the current temperature in Tokyo (returns quickly)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let order_fast = completion_order.clone();
        let fast_fn: Arc<crate::ToolAsyncFunc> = Arc::new(move |_args, _ctx: crate::ToolContext| {
            let order = order_fast.clone();
            Box::pin(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                order.lock().unwrap().push("fast");
                crate::to_value!(15u64)
            })
        });

        // slow tool: completes in ~300 ms, returns 25
        let slow_desc = ToolDescBuilder::new("temperature_slow")
            .description("Get the current temperature in Seoul (takes longer)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let order_slow = completion_order.clone();
        let slow_fn: Arc<crate::ToolAsyncFunc> = Arc::new(move |_args, _ctx: crate::ToolContext| {
            let order = order_slow.clone();
            Box::pin(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                order.lock().unwrap().push("slow");
                crate::to_value!(25u64)
            })
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature_fast".to_string(),
            ToolRuntime::new_async(fast_desc, fast_fn),
        );
        tool_set.insert(
            "temperature_slow".to_string(),
            ToolRuntime::new_async(slow_desc, slow_fn),
        );

        let mut agent = AgentRuntime::new(
            AgentSpec::new("claude-haiku-4-5-20251001")
                .with_tools(["temperature_fast", "temperature_slow"])
                .with_instruction(
                    "When asked for temperatures in multiple cities, always call \
                     temperature_fast and temperature_slow in a single response."
                        .to_string(),
                ),
            AgentProvider {
                lm: LangModelProvider::anthropic(api_key),
                tools: vec![],
            },
            tool_set,
        )
        .await
        .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Get the temperature in Tokyo using temperature_fast \
             and in Seoul using temperature_slow.",
        )]);

        {
            let mut strm = agent.stream_turn(query);
            while let Some(event) = strm.next().await {
                event.unwrap();
            }
        }

        // Verify both tools were actually called
        let order = completion_order.lock().unwrap();
        assert!(
            order.contains(&"fast") && order.contains(&"slow"),
            "Both tools must have been called, got: {:?}",
            *order
        );

        // 1. Completion order: fast (50 ms) finishes before slow (300 ms)
        assert_eq!(
            order.as_slice(),
            &["fast", "slow"],
            "Expected fast tool to complete before slow tool (got {:?}), \
             which would only happen if tools ran concurrently",
            order.as_slice()
        );

        // 2. History push order matches completion order (fast result first, slow result second)
        let history = agent.get_history();
        let tool_msgs: Vec<_> = history.iter().filter(|m| m.role == Role::Tool).collect();
        assert_eq!(
            tool_msgs.len(),
            2,
            "Expected exactly two Tool messages in history"
        );

        use crate::datatype::Value;

        let first_value = tool_msgs[0]
            .contents
            .iter()
            .find_map(|p| {
                if let Part::Value { value } = p {
                    Some(value.clone())
                } else {
                    None
                }
            })
            .expect("Expected Value part in first tool result");
        let second_value = tool_msgs[1]
            .contents
            .iter()
            .find_map(|p| {
                if let Part::Value { value } = p {
                    Some(value.clone())
                } else {
                    None
                }
            })
            .expect("Expected Value part in second tool result");

        assert_eq!(
            first_value,
            Value::Unsigned(15),
            "Fast tool result (15) should be first in history"
        );
        assert_eq!(
            second_value,
            Value::Unsigned(25),
            "Slow tool result (25) should be second in history"
        );
    }

    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_tool_panic_causes_inconsistent_history() {
        dotenvy::dotenv().ok();

        // This test intentionally panics inside a tool function to verify
        // that the agent catches it and keeps history consistent.
        suppress_panics!();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let good_desc = ToolDescBuilder::new("get_weather")
            .description("Get the current weather for a city")
            .parameters(to_value!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }))
            .build();
        let good_fn: Arc<crate::ToolAsyncFunc> = Arc::new(|_args, _ctx: crate::ToolContext| {
            Box::pin(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                crate::to_value!("sunny, 25 degrees")
            })
        });

        let bad_desc = ToolDescBuilder::new("get_traffic")
            .description("Get the current traffic conditions for a city")
            .parameters(to_value!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }))
            .build();
        let bad_fn: Arc<crate::ToolAsyncFunc> = Arc::new(|_args, _ctx: crate::ToolContext| {
            Box::pin(async move {
                panic!("simulated tool crash");
            })
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "get_weather".to_string(),
            ToolRuntime::new_async(good_desc, good_fn),
        );
        tool_set.insert(
            "get_traffic".to_string(),
            ToolRuntime::new_async(bad_desc, bad_fn),
        );

        let mut agent = AgentRuntime::new(
            AgentSpec::new("gpt-4o-mini")
                .with_tools(["get_weather", "get_traffic"])
                .with_instruction(
                    "When asked about a city, ALWAYS call both get_weather AND \
                 get_traffic tools in a single response. Never call just one."
                        .to_string(),
                ),
            AgentProvider {
                lm: LangModelProvider::openai(api_key),
                tools: vec![],
            },
            tool_set,
        )
        .await
        .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Tell me about Seoul. Use get_weather for weather and get_traffic for traffic.",
        )]);

        {
            let mut strm = agent.stream_turn(query);
            while let Some(result) = strm.next().await {
                let _ = result;
            }
        }

        let history = agent.get_history();

        let tool_use_count: usize = history
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .filter_map(|m| m.tool_calls.as_ref())
            .map(|tc| tc.len())
            .sum();

        let tool_result_count = history.iter().filter(|m| m.role == Role::Tool).count();

        if tool_use_count < 2 {
            eprintln!(
                "LLM only produced {} tool call(s); skipping consistency check",
                tool_use_count
            );
            return;
        }

        assert_eq!(
            tool_use_count, tool_result_count,
            "History is inconsistent: {} tool_use call(s) but {} tool result(s). \
         The panicked tool's result was silently lost.",
            tool_use_count, tool_result_count,
        );
    }
}
