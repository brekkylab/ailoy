use std::pin::Pin;

use futures::{Stream, StreamExt as _};

use crate::{
    agent::{AgentProvider, AgentSpec},
    lang_model::LangModelRuntime,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    shell::Shell,
    tool::{ToolRuntime, ToolSet},
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
    pub state: AgentState,
}

impl AgentRuntime {
    /// Return the full message history accumulated so far.
    pub fn get_history(&self) -> &[Message] {
        &self.state.history
    }

    pub async fn try_new(spec: AgentSpec, provider: AgentProvider) -> anyhow::Result<Self> {
        // Initialize tool set
        let tool_set = ToolSet::from_providers(&provider).await?;
        Self::try_from_toolset(spec, provider, &tool_set)
    }

    pub fn try_from_toolset(
        spec: AgentSpec,
        provider: AgentProvider,
        tool_set: &ToolSet,
    ) -> anyhow::Result<Self> {
        // Resolve LM provider
        let lm_provider = provider
            .get_model(&spec.model)
            .ok_or_else(|| anyhow::anyhow!("No provider found for model '{}'", spec.model))?
            .clone();

        // Collect tools from tool_set; error if any tool is missing
        let tools: Vec<ToolRuntime> = spec
            .tools
            .iter()
            .map(|n| {
                tool_set
                    .make_runtime(n)
                    .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found in tool_set", n))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Initialize history with system instruction if present
        let history = spec
            .instruction
            .as_ref()
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        Ok(Self {
            lm: LangModelRuntime::new(spec.model.clone(), lm_provider),
            tools,
            state: AgentState::with_history(history),
        })
    }

    /// Stream all events for a single agent turn.
    pub fn run(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageOutput>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            self.state.history.push(query);
            let tool_descs: Vec<_> = self.tools.iter().map(|t| t.get_desc().clone()).collect();

            loop {
                let mut output = self.lm.run(&self.state.history, &tool_descs).await?;
                output.depth = Some(0);
                self.state.history.push(output.message.clone());

                let tool_calls = match &output.finish_reason {
                    FinishReason::ToolCall {} => {
                        let tc = output.message.tool_calls.clone().unwrap_or_default();
                        yield output;
                        tc
                    },
                    _ => {
                        yield output;
                        break;
                    }
                };

                // Execute tool calls sequentially.
                for tool_call in tool_calls {
                    let tool_name = match tool_call.as_function() {
                        Some((_, name, _)) => name.to_string(),
                        None => continue,
                    };

                    let func = match self.tools.iter().find(|t| t.get_desc().name == tool_name) {
                        Some(t) => t.get_func(),
                        None => Err(anyhow::anyhow!("No tool found for '{}'", tool_name))?,
                    };

                    let mut stream = func.call(tool_call)?;

                    // Buffer one item so we can push the last one to history.
                    // Intermediate outputs are tool-internal messages (depth + 1).
                    // The final output is the tool response at the current depth (0).
                    let mut last: Option<MessageOutput> = None;
                    while let Some(item) = stream.next().await {
                        if let Some(mut prev) = last.replace(item) {
                            prev.depth = Some(prev.depth.map_or(0, |d| d) + 1);
                            yield prev;
                        }
                    }

                    match last {
                        Some(mut item) => {
                            item.depth = None;
                            self.state.history.push(item.message.clone());
                            yield item;
                        }
                        None => Err(anyhow::anyhow!("tool '{}' produced no output", tool_name))?,
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt as _;
    use tokio::sync::Mutex;

    use super::*;
    use crate::{
        agent::{AgentProvider, AgentSpec},
        datatype::Value,
        message::{AgentCard, FinishReason, Message, Part, Role, ToolDesc, ToolDescBuilder},
        suppress_panics, to_value,
        tool::{ToolFunc, ToolSet},
        tool_impl::make_subagent_tool,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    fn temperature_tool_desc() -> ToolDesc {
        ToolDescBuilder::new("temperature")
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
            .build()
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_agent() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature",
            temperature_tool_desc(),
            ToolFunc::new(|_args: Value| Value::unsigned(25)),
        );

        let provider = AgentProvider::new().model_openai(api_key);
        let spec = AgentSpec::new("openai/gpt-4o-mini").tool("temperature");
        let mut agent = AgentRuntime::try_from_toolset(spec, provider, &tool_set).unwrap();

        let query = Message::new(Role::User)
            .with_contents([Part::text("What is the current temperature in Seoul?")]);

        let mut strm = agent.run(query);
        let mut events = vec![];
        while let Some(event) = strm.next().await {
            events.push(event.unwrap());
        }

        // The last assistant Stop message is the final answer.
        let final_output = events
            .iter()
            .rev()
            .find(|e| {
                e.message.role == Role::Assistant
                    && matches!(e.finish_reason, FinishReason::Stop {})
            })
            .expect("Expected a final assistant message");

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
        let provider = AgentProvider::new().model_openai(api_key.clone());

        // Sub-agent: a minimal calculator that replies with just the numeric result.
        let sub_spec = AgentSpec::new("openai/gpt-4o-mini").instruction(
            "You are a calculator. Answer math questions with the numeric result only.".to_string(),
        );
        let sub_agent =
            AgentRuntime::try_from_toolset(sub_spec, provider.clone(), &ToolSet::new()).unwrap();
        let sub_agent = Arc::new(Mutex::new(sub_agent));

        let card = AgentCard {
            name: "math-agent".to_string(),
            description:
                "Handles arithmetic and math computations. Use this for any math question."
                    .to_string(),
            skills: vec![],
        };
        let sub_tool = make_subagent_tool(card, sub_agent);

        // Main agent: coordinator that should always delegate math to math-agent.
        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "math-agent",
            sub_tool.get_desc().clone(),
            sub_tool.get_func(),
        );

        let mut main_agent = AgentRuntime::try_from_toolset(
            AgentSpec::new("openai/gpt-4o-mini").tool("math-agent"),
            provider,
            &tool_set,
        )
        .unwrap();

        let query =
            Message::new(Role::User).with_contents([Part::text("What is 123 multiplied by 7?")]);

        {
            let mut strm = main_agent.run(query);
            while let Some(event) = strm.next().await {
                event.unwrap();
            }
        }

        // The history must contain a Tool message, confirming the subagent was called.
        let history = main_agent.get_history();
        assert!(
            history.iter().any(|m| m.role == Role::Tool),
            "Expected main agent history to contain a Tool message (subagent was called)"
        );

        // The final assistant message must contain text.
        let last_assistant = history
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .expect("Expected at least one assistant message");
        assert!(
            last_assistant.contents.iter().any(|p| p.is_text()),
            "Expected final assistant message to contain text"
        );
    }

    /// Verifies that run() emits intermediate sub-agent outputs (depth > 0)
    /// followed by a final Role::Tool result (depth == 0) when using a streaming
    /// subagent tool.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_streaming_subagent_emits_tool_deltas() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");
        let provider = AgentProvider::new().model_openai(api_key.clone());

        // Sub-agent: gives a multi-step response so we see intermediate messages.
        let sub_spec = AgentSpec::new("openai/gpt-4o-mini").instruction(
            "You are a calculator. Answer math questions with the numeric result only.".to_string(),
        );
        let sub_agent =
            AgentRuntime::try_from_toolset(sub_spec, provider.clone(), &ToolSet::new()).unwrap();
        let sub_agent = Arc::new(Mutex::new(sub_agent));

        let card = AgentCard {
            name: "math-agent".to_string(),
            description: "Handles arithmetic and math computations.".to_string(),
            skills: vec![],
        };
        let sub_tool = make_subagent_tool(card, sub_agent);

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "math-agent",
            sub_tool.get_desc().clone(),
            sub_tool.get_func(),
        );

        let mut main_agent = AgentRuntime::try_from_toolset(
            AgentSpec::new("openai/gpt-4o-mini").tool("math-agent"),
            provider,
            &tool_set,
        )
        .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text("What is 99 plus 1?")]);

        let mut strm = main_agent.run(query);
        let mut tool_deltas = 0usize;
        let mut tool_results = 0usize;

        while let Some(event) = strm.next().await {
            let output = event.unwrap();
            // Intermediate sub-agent outputs: depth > 0.
            if output.depth.is_some_and(|d| d > 0) {
                tool_deltas += 1;
            }
            // Final tool result yielded back to the outer agent: Role::Tool at depth 0.
            if output.message.role == Role::Tool && output.depth == Some(0) {
                tool_results += 1;
            }
        }

        assert!(tool_results > 0, "Expected at least one tool result");
        assert!(
            tool_deltas > 0,
            "Expected at least one intermediate sub-agent output (tool delta)"
        );
    }

    /// Verifies event sequence for a simple tool call:
    /// AssistantMessage(ToolCall) -> Tool result -> AssistantMessage(Stop)
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_turn_event_sequence() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature",
            temperature_tool_desc(),
            ToolFunc::new(|_args: Value| Value::unsigned(25)),
        );

        let provider = AgentProvider::new().model_openai(api_key);
        let spec = AgentSpec::new("openai/gpt-4o-mini").tool("temperature");
        let mut agent = AgentRuntime::try_from_toolset(spec, provider, &tool_set).unwrap();

        let query = Message::new(Role::User)
            .with_contents([Part::text("What is the temperature in Seoul?")]);

        let mut strm = agent.run(query);
        let mut events = vec![];
        while let Some(event) = strm.next().await {
            events.push(event.unwrap());
        }

        // Must contain at least one ToolCall assistant message.
        let has_tool_call = events
            .iter()
            .any(|e| e.message.role == Role::Assistant && e.message.tool_calls.is_some());
        assert!(
            has_tool_call,
            "Expected an assistant message with tool calls"
        );

        // Must contain at least one tool result.
        let has_tool_result = events.iter().any(|e| e.message.role == Role::Tool);
        assert!(has_tool_result, "Expected a tool result message");

        // Last event must be a final assistant text response.
        let last = events.last().expect("Expected at least one event");
        assert_eq!(last.message.role, Role::Assistant);
        assert!(
            last.message.contents.iter().any(|p| p.is_text()),
            "Last event should contain text"
        );
    }

    /// Verifies that multiple tool calls in a single LLM response are executed
    /// and both results appear in history.
    ///
    /// Two async tools are registered:
    /// - `temperature_fast`: 50 ms delay, returns 15
    /// - `temperature_slow`: 300 ms delay, returns 25
    ///
    /// Note: tool calls are currently executed sequentially, so this test
    /// verifies correct sequential ordering rather than concurrency.
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_parallel_tool_calls() {
        dotenvy::dotenv().ok();

        use std::sync::{Arc as StdArc, Mutex as StdMutex};

        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in .env");

        // Record the order in which tools complete.
        let completion_order: StdArc<StdMutex<Vec<&'static str>>> =
            StdArc::new(StdMutex::new(Vec::new()));

        let order_fast = completion_order.clone();
        let fast_desc = ToolDescBuilder::new("temperature_fast")
            .description("Get the current temperature in Tokyo (returns quickly)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let fast_fn = ToolFunc::new(move |_args: Value| {
            let order = order_fast.clone();
            async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                order.lock().unwrap().push("fast");
                to_value!(15u64)
            }
        });

        let order_slow = completion_order.clone();
        let slow_desc = ToolDescBuilder::new("temperature_slow")
            .description("Get the current temperature in Seoul (takes longer)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let slow_fn = ToolFunc::new(move |_args: Value| {
            let order = order_slow.clone();
            async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                order.lock().unwrap().push("slow");
                to_value!(25u64)
            }
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert("temperature_fast", fast_desc, fast_fn);
        tool_set.insert("temperature_slow", slow_desc, slow_fn);

        let provider = AgentProvider::new().model_claude(api_key);
        let mut agent = AgentRuntime::try_from_toolset(
            AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
                .tools(["temperature_fast", "temperature_slow"])
                .instruction(
                    "When asked for temperatures in multiple cities, always call \
                     temperature_fast and temperature_slow in a single response."
                        .to_string(),
                ),
            provider,
            &tool_set,
        )
        .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Get the temperature in Tokyo using temperature_fast \
             and in Seoul using temperature_slow.",
        )]);

        {
            let mut strm = agent.run(query);
            while let Some(event) = strm.next().await {
                event.unwrap();
            }
        }

        // Verify both tools were actually called.
        let order = completion_order.lock().unwrap();
        assert!(
            order.contains(&"fast") && order.contains(&"slow"),
            "Both tools must have been called, got: {:?}",
            *order
        );

        // With sequential execution fast is called first, so it finishes first.
        assert_eq!(
            order.as_slice(),
            &["fast", "slow"],
            "Expected fast to complete before slow (sequential order): {:?}",
            order.as_slice()
        );

        // History must contain results for both tools.
        let history = agent.get_history();
        let tool_msgs: Vec<_> = history.iter().filter(|m| m.role == Role::Tool).collect();
        assert_eq!(
            tool_msgs.len(),
            2,
            "Expected exactly two Tool messages in history"
        );

        // First result is 15 (fast), second is 25 (slow).
        let first_value = tool_msgs[0]
            .contents
            .iter()
            .find_map(|p| p.as_value().cloned())
            .expect("Expected Value part in first tool result");
        let second_value = tool_msgs[1]
            .contents
            .iter()
            .find_map(|p| p.as_value().cloned())
            .expect("Expected Value part in second tool result");

        assert_eq!(
            first_value,
            Value::unsigned(15),
            "Fast tool result should be 15"
        );
        assert_eq!(
            second_value,
            Value::unsigned(25),
            "Slow tool result should be 25"
        );
    }

    /// Verifies history consistency when one of two parallel tool calls panics.
    ///
    /// Two tools are registered — one good, one that panics — and the model is
    /// instructed to call both in the same response.  The assertion checks that
    /// the number of tool-call entries in history equals the number of tool-result
    /// entries, so that history is never left in a half-written state.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_tool_panic_causes_inconsistent_history() {
        dotenvy::dotenv().ok();

        // This test intentionally panics inside a tool function.
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
        let good_fn = ToolFunc::new(|_args: Value| async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            to_value!("sunny, 25 degrees")
        });

        let bad_desc = ToolDescBuilder::new("get_traffic")
            .description("Get the current traffic conditions for a city")
            .parameters(to_value!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }))
            .build();
        let bad_fn = ToolFunc::new(|_args: Value| async move {
            panic!("simulated tool crash");
            #[allow(unreachable_code)]
            to_value!("unreachable")
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert("get_weather", good_desc, good_fn);
        tool_set.insert("get_traffic", bad_desc, bad_fn);

        let provider = AgentProvider::new().model_openai(api_key);
        let mut agent = AgentRuntime::try_from_toolset(
            AgentSpec::new("openai/gpt-4o-mini")
                .tools(["get_weather", "get_traffic"])
                .instruction(
                    "When asked about a city, ALWAYS call both get_weather AND \
                     get_traffic tools in a single response. Never call just one."
                        .to_string(),
                ),
            provider,
            &tool_set,
        )
        .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Tell me about Seoul. Use get_weather for weather and get_traffic for traffic.",
        )]);

        {
            let mut strm = agent.run(query);
            while let Some(result) = strm.next().await {
                let _ = result; // ignore errors produced by the panic
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
