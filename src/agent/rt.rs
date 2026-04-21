use std::{collections::HashMap, pin::Pin, sync::Arc};

use futures::{FutureExt as _, Stream, StreamExt as _};

use crate::{
    agent::{AgentProvider, AgentSpec, default_provider},
    lang_model::LangModel,
    message::{FinishReason, Message, MessageOutput, Part, Role, ToolDesc},
    tool::{Tool, ToolFunc, ToolSet},
};

pub struct AgentState {
    pub history: Vec<Message>,

    #[cfg(feature = "sandbox-microvm")]
    pub sandbox: Option<std::sync::Arc<crate::sandbox::Sandbox>>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            #[cfg(feature = "sandbox-microvm")]
            sandbox: None,
        }
    }

    pub fn with_history(history: Vec<Message>) -> Self {
        Self {
            history,
            #[cfg(feature = "sandbox-microvm")]
            sandbox: None,
        }
    }
}

/// An agent that drives a language model through multi-turn, tool-augmented conversations.
///
/// `Agent` pairs an [`AgentSpec`] (model + instruction + tools + sub-agents) with an
/// [`AgentProvider`] (credentials + tool sources) and an internal [`AgentState`]
/// (message history).  Call [`Agent::run`] to stream a single turn; tool calls are
/// resolved automatically and the conversation is appended to history after each turn.
///
/// For construction examples, see [`Agent::try_new`], [`Agent::try_with_provider`], and
/// [`Agent::try_with_tools`].
pub struct Agent {
    model: LangModel,
    tools: Vec<Tool>,
    pub state: AgentState,
}

impl Agent {
    /// Create an agent.
    ///
    /// Uses the process-wide [`default_provider`] for configuration. Configure it once
    /// at startup; all agents built with this method share those credentials and tool
    /// sources without passing a provider around.
    ///
    /// ```rust
    /// # use ailoy::{agent::{Agent, AgentSpec, default_provider_mut}, message::{Message, Part, Role}};
    /// # use futures::StreamExt as _;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// // One-time setup — configure the global provider before creating any agents.
    /// default_provider_mut().await.model_claude("ANTHROPIC_API_KEY");
    ///
    /// let spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001");
    /// let agent = Agent::try_new(spec).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        let provider = default_provider().await;
        Self::try_with_provider(spec, &provider).await
    }

    /// Create an agent with an explicit [`AgentProvider`].
    ///
    /// Use this for scoped, explicit control over models and tool sources without
    /// touching global state.  The provider is not stored in the agent and can be
    /// reused across multiple agents.
    ///
    /// ```rust
    /// # use ailoy::agent::{Agent, AgentProvider, AgentSpec};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let mut provider = AgentProvider::new();
    /// provider.model_openai("OPENAI_API_KEY").tool_web_search();
    ///
    /// let spec = AgentSpec::new("openai/gpt-4o").tool("web_search");
    /// let agent = Agent::try_with_provider(spec, &provider).await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub async fn try_with_provider(
        spec: AgentSpec,
        provider: &AgentProvider,
    ) -> anyhow::Result<Self> {
        let tools = ToolSet::from_providers(&spec, provider).await?;
        Self::try_with_tools(spec, provider, &tools).await
    }

    /// Create an agent with a pre-built toolset, bypassing automatic tool-source initialisation.
    ///
    /// Use this when you need deterministic, in-process tools (e.g. unit tests, mock tools,
    /// or tools assembled at runtime from a [`ToolSet`]).
    ///
    /// ```rust
    /// # use ailoy::{agent::{Agent, AgentProvider, AgentSpec}, datatype::Value, message::ToolDescBuilder, to_value, tool::{ToolFunc, ToolSet}};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    ///     let mut tool_set = ToolSet::new();
    ///     tool_set.insert(
    ///         "temperature",
    ///         ToolDescBuilder::new("temperature")
    ///             .description("Return the temperature for a city")
    ///             .parameters(to_value!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}))
    ///             .build(),
    ///         ToolFunc::new(|_args: Value| Value::unsigned(25)),
    ///     );
    ///
    ///     let mut provider = AgentProvider::new();
    ///     provider.model_openai("OPENAI_API_KEY");
    ///     let spec = AgentSpec::new("openai/gpt-4o-mini").tool("temperature");
    ///     let agent = Agent::try_with_tools(spec, &provider, &tool_set).await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub async fn try_with_tools(
        spec: AgentSpec,
        provider: &AgentProvider,
        tools: impl IntoIterator<Item = (ToolDesc, Arc<ToolFunc>)>,
    ) -> anyhow::Result<Self> {
        // Parse model id
        let model_id = spec
            .model
            .split_once('/')
            .map(|(_, id)| id.to_string())
            .unwrap_or_else(|| spec.model.clone());
        // Resolve LangModel provider
        let model_provider = provider
            .get_model(&spec.model)
            .ok_or_else(|| anyhow::anyhow!("No provider found for model '{}'", spec.model))?
            .clone();

        // Build a name-keyed map from the provided tools
        let tool_map: HashMap<String, (ToolDesc, Arc<ToolFunc>)> = tools
            .into_iter()
            .map(|(desc, f)| (desc.name.clone(), (desc, f)))
            .collect();

        // Collect tools required by the spec; error if any tool is missing
        let tools: Vec<Tool> = spec
            .tools
            .iter()
            .map(|n| {
                tool_map
                    .get(n)
                    .map(|(desc, f)| Tool::new(desc.clone(), Arc::clone(f)))
                    .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", n))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Initialize history with system instruction if present
        let history = spec
            .instruction
            .as_ref()
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        #[allow(unused_mut)]
        let mut state = AgentState::with_history(history);
        #[cfg(feature = "sandbox-microvm")]
        {
            state.sandbox = provider.sandbox.clone();
        }
        Ok(Self {
            model: LangModel::new(model_id, model_provider),
            tools,
            state,
        })
    }

    /// Execute tool calls in parallel and return a stream of all outputs.
    ///
    /// Synchronously spawns one tokio task per tool call, then returns a stream
    /// backed by an mpsc channel. Each task runs the tool, forwards intermediate
    /// sub-agent outputs (with bumped depth), and emits a final `Role::Tool`
    /// message at `depth = Some(0)`. Panics are caught via `catch_unwind` and
    /// converted to synthetic error tool results so the LM always receives
    /// exactly one result per call.
    ///
    /// Returns `Err` immediately (before spawning) if any tool name is not found.
    fn execute_tool_calls(
        &self,
        tool_calls: Vec<Part>,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<MessageOutput>>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<MessageOutput>>();

        for tool_call in tool_calls {
            let Some((call_id, tool_name, _)) = tool_call.as_function() else {
                continue;
            };
            let (tool_name, call_id) = (tool_name.to_string(), call_id.to_string());

            let func = self
                .tools
                .iter()
                .find(|t| t.get_desc().name == tool_name)
                .map(|t| t.get_func())
                .ok_or_else(|| anyhow::anyhow!("No tool found for '{}'", tool_name))?;

            let tx = tx.clone();

            tokio::spawn(async move {
                // tx_inner is moved into AssertUnwindSafe and may be dropped
                // during a panic unwind. tx stays alive for the fallback send.
                let tx_inner = tx.clone();
                let tool_name_inner = tool_name.clone();

                let outcome: Result<anyhow::Result<bool>, _> =
                    std::panic::AssertUnwindSafe(async move {
                        let mut stream = func.call(tool_call)?;
                        let mut last: Option<MessageOutput> = None;

                        while let Some(item) = stream.next().await {
                            if let Some(mut prev) = last.replace(item) {
                                prev.depth = Some(prev.depth.map_or(0, |d| d) + 1);
                                if tx_inner.send(Ok(prev)).is_err() {
                                    return anyhow::Ok(false);
                                }
                            }
                        }

                        match last {
                            Some(mut item) => {
                                item.depth = Some(0);
                                item.message.role = Role::Tool;
                                let _ = tx_inner.send(Ok(item));
                                anyhow::Ok(true)
                            }
                            None => anyhow::Ok(false),
                        }
                    })
                    .catch_unwind()
                    .await;

                let needs_fallback = !matches!(outcome, Ok(Ok(true)));
                if needs_fallback {
                    if let Ok(Err(e)) = outcome {
                        let _ = tx.send(Err(e));
                    } else {
                        let reason = if outcome.is_err() {
                            "panicked during execution"
                        } else {
                            "produced no output"
                        };
                        let err_msg = Message::new(Role::Tool)
                            .with_contents([Part::value(crate::datatype::Value::string(format!(
                                "tool '{}' {}",
                                tool_name_inner, reason
                            )))])
                            .with_id(call_id);
                        let _ = tx.send(Ok(MessageOutput {
                            depth: Some(0),
                            message: err_msg,
                            finish_reason: FinishReason::Stop {},
                        }));
                    }
                }
            });
        }

        // Drop the original sender so the channel closes once all tasks finish.
        drop(tx);

        Ok(Box::pin(async_stream::stream! {
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                yield event;
            }
        }))
    }

    /// Return the full message history accumulated so far.
    pub fn get_history(&self) -> &[Message] {
        &self.state.history
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
                let mut output = self.model.run(&self.state.history, &tool_descs).await?;
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

                let mut tool_stream = self.execute_tool_calls(tool_calls)?;
                while let Some(event) = tool_stream.next().await {
                    match event {
                        Err(e) => Err(e)?,
                        Ok(output) => {
                            if output.message.role == Role::Tool && output.depth == Some(0) {
                                self.state.history.push(output.message.clone());
                            }
                            yield output;
                        }
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
        agent::{AgentCard, AgentProvider, AgentSpec},
        datatype::Value,
        message::{Message, Part, Role, ToolDesc, ToolDescBuilder},
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
    async fn test_simple_tool_call() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature",
            temperature_tool_desc(),
            ToolFunc::new(|_args: Value| Value::unsigned(25)),
        );

        let mut provider = AgentProvider::new();
        provider.model_openai(api_key);
        let spec = AgentSpec::new("openai/gpt-4o-mini").tool("temperature");
        let mut agent = Agent::try_with_tools(spec, &provider, &tool_set)
            .await
            .unwrap();

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

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    /// Verifies that the main agent actually delegates to the in-memory subagent.
    ///
    /// Sets up a math subagent and registers it as a subagent tool on a coordinator agent.
    /// Asks a math question and confirms the main agent's history contains a [`Role::Tool`]
    /// message (proof that the subagent tool was called), and that the final reply is text.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_delegate_to_subagent() {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");
        let mut provider = AgentProvider::new();
        provider.model_openai(api_key.clone());

        // Sub-agent: a minimal calculator that replies with just the numeric result.
        let sub_spec = AgentSpec::new("openai/gpt-4o-mini").instruction(
            "You are a calculator. Answer math questions with the numeric result only.".to_string(),
        );
        let sub_agent = Agent::try_with_tools(sub_spec, &provider, &ToolSet::new())
            .await
            .unwrap();
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

        let mut main_agent = Agent::try_with_tools(
            AgentSpec::new("openai/gpt-4o-mini")
                .tool("math-agent")
                .instruction(
                    "You are a coordinator. For any arithmetic or math question, \
                     always delegate to the math-agent tool."
                        .to_string(),
                ),
            &provider,
            &tool_set,
        )
        .await
        .unwrap();

        let query =
            Message::new(Role::User).with_contents([Part::text("What is 123 multiplied by 7?")]);

        {
            let mut strm = main_agent.run(query);
            while let Some(event) = strm.next().await {
                let event = event.unwrap();
                println!("{}", event);
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
        let mut provider = AgentProvider::new();
        provider.model_openai(api_key.clone());

        // Sub-agent: gives a multi-step response so we see intermediate messages.
        let sub_spec = AgentSpec::new("openai/gpt-4o-mini").instruction(
            "You are a calculator. Answer math questions with the numeric result only.".to_string(),
        );
        let sub_agent = Agent::try_with_tools(sub_spec, &provider, &ToolSet::new())
            .await
            .unwrap();
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

        let mut main_agent = Agent::try_with_tools(
            AgentSpec::new("openai/gpt-4o-mini").tool("math-agent"),
            &provider,
            &tool_set,
        )
        .await
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

        let mut provider = AgentProvider::new();
        provider.model_claude(api_key);
        let mut agent = Agent::try_with_tools(
            AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
                .tools(["temperature_fast", "temperature_slow"])
                .instruction(
                    "When asked for temperatures in multiple cities, always call \
                     temperature_fast and temperature_slow in a single response."
                        .to_string(),
                ),
            &provider,
            &tool_set,
        )
        .await
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
        let bad_fn = ToolFunc::new(|_args: Value| async move { panic!("simulated tool crash") });

        let mut tool_set = ToolSet::new();
        tool_set.insert("get_weather", good_desc, good_fn);
        tool_set.insert("get_traffic", bad_desc, bad_fn);

        let mut provider = AgentProvider::new();
        provider.model_openai(api_key);
        let mut agent = Agent::try_with_tools(
            AgentSpec::new("openai/gpt-4o-mini")
                .tools(["get_weather", "get_traffic"])
                .instruction(
                    "When asked about a city, ALWAYS call both get_weather AND \
                     get_traffic tools in a single response. Never call just one."
                        .to_string(),
                ),
            &provider,
            &tool_set,
        )
        .await
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

        // 1. Every tool call must have a corresponding tool result — no silent drops.
        assert_eq!(
            tool_use_count, tool_result_count,
            "History is inconsistent: {} tool_use call(s) but {} tool result(s). \
             The panicked tool's result was silently lost.",
            tool_use_count, tool_result_count,
        );

        // 2. All tool result messages must use Part::Value, not Part::Text.
        //    Part::Text is marshalled as a JSON object by provider codecs, which
        //    breaks the `function_call_output.output` field (must be a string).
        for tool_msg in history.iter().filter(|m| m.role == Role::Tool) {
            for part in &tool_msg.contents {
                assert!(
                    part.as_value().is_some(),
                    "Tool result content must be Part::Value for correct API marshalling, \
                     but found a non-Value part. This causes provider API errors when the \
                     result is sent back to the model."
                );
            }
        }

        // 3. History must end with a final Assistant text response, not a tool call.
        //    If the second LM call fails (e.g. due to a malformed tool result), the
        //    stream terminates early and the agent never produces a closing answer.
        let last_msg = history.last().expect("History should not be empty");
        assert_eq!(
            last_msg.role,
            Role::Assistant,
            "History must end with an Assistant message, not {:?}",
            last_msg.role
        );
        assert!(
            last_msg.contents.iter().any(|p| p.is_text()),
            "Final Assistant message must contain text — the agent never produced a closing answer."
        );
    }
}
