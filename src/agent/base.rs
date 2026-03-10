use std::sync::Arc;

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    knowledge::{Knowledge, KnowledgeDyn},
    model::{LangModel, LangModelInferConfig, LangModelInference as _},
    tool::{Tool, ToolBehavior as _},
    utils::{BoxStream, log},
    value::{
        Delta, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part,
        PartDelta, Role, ToolDesc,
    },
};

/// Configuration for running the agent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference: Option<LangModelInferConfig>,
}

/// Builder for constructing an `Agent`.
pub struct AgentBuilder {
    model: Option<LangModel>,
    tools: Vec<Tool>,
    system: Option<String>,
    knowledge: Option<Arc<KnowledgeDyn>>,
}

impl AgentBuilder {
    fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            system: None,
            knowledge: None,
        }
    }

    pub fn model(mut self, m: LangModel) -> Self {
        self.model = Some(m);
        self
    }

    pub fn tool(mut self, t: Tool) -> Self {
        self.tools.push(t);
        self
    }

    pub fn tools(mut self, ts: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.extend(ts);
        self
    }

    pub fn system(mut self, s: impl Into<String>) -> Self {
        self.system = Some(s.into());
        self
    }

    /// Attach a knowledge source for automatic per-turn retrieval (Pattern B).
    /// At most one knowledge source is supported; call this once.
    pub fn knowledge(mut self, k: impl Knowledge + 'static) -> Self {
        self.knowledge = Some(Arc::new(k));
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            lm: self.model.expect("AgentBuilder: model is required"),
            tools: self.tools,
            system: self.system,
            history: Vec::new(),
            knowledge: self.knowledge,
        }
    }
}

/// The Agent orchestrates language model, tools, and the reasoning loop.
#[derive(Clone)]
pub struct Agent {
    lm: LangModel,
    tools: Vec<Tool>,
    system: Option<String>,
    history: Vec<Message>,
    /// Optional knowledge source for automatic per-turn retrieval (Pattern B).
    knowledge: Option<Arc<KnowledgeDyn>>,
}

impl Agent {
    pub fn new(lm: LangModel, tools: impl IntoIterator<Item = Tool>) -> Self {
        Self {
            lm,
            tools: tools.into_iter().collect(),
            system: None,
            history: Vec::new(),
            knowledge: None,
        }
    }

    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    pub fn get_lm(&self) -> LangModel {
        self.lm.clone()
    }

    pub fn get_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn add_tools(&mut self, tools: Vec<Tool>) {
        for tool in tools.iter() {
            let tool_name = tool.get_description().name.clone();
            if self
                .tools
                .iter()
                .any(|t| t.get_description().name == tool_name)
            {
                log::warn(format!(
                    "Tool \"{}\" is already registered. Skip adding.",
                    tool_name
                ));
                continue;
            }
            self.tools.push(tool.clone());
        }
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.add_tools(vec![tool]);
    }

    pub fn remove_tools(&mut self, tool_names: Vec<String>) {
        self.tools
            .retain(|t| !tool_names.contains(&t.get_description().name));
    }

    pub fn remove_tool(&mut self, tool_name: String) {
        self.remove_tools(vec![tool_name]);
    }

    pub fn clear_tools(&mut self) {
        self.tools.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Sync setup shared by every public turn method:
    /// - builds system + history base messages
    /// - pushes the clean user message into history
    /// - returns owned copies of the data needed inside the async stream body
    fn prepare_turn(
        &mut self,
        prompt: String,
        config: Option<AgentConfig>,
    ) -> (Message, Vec<Message>, Vec<Tool>, LangModelInferConfig) {
        let clean_user_msg = Message::new(Role::User).with_contents([Part::text(prompt.clone())]);
        let mut base_messages = Vec::new();
        if let Some(system) = &self.system {
            base_messages
                .push(Message::new(Role::System).with_contents([Part::text(system.clone())]));
        }
        base_messages.extend(self.history.clone());
        self.history.push(clean_user_msg.clone());
        let tools = self.tools.clone();
        let inference_config = config.unwrap_or_default().inference.unwrap_or_default();
        (clean_user_msg, base_messages, tools, inference_config)
    }

    /// Retrieve context from the attached knowledge source for the given query.
    /// Returns `None` if no knowledge source is configured or no documents were found.
    async fn retrieve_context(&self, query: &str) -> anyhow::Result<Option<String>> {
        let Some(k) = &self.knowledge else {
            return Ok(None);
        };
        let docs = k.retrieve(query).await?;
        if docs.is_empty() {
            Ok(None)
        } else {
            Ok(Some(k.format_context(query, &docs)))
        }
    }

    /// Perform retrieval and assemble the full message list for the LLM.
    /// The returned list ends with the (possibly context-augmented) user message.
    async fn build_llm_messages(
        &self,
        prompt: &str,
        clean_user_msg: Message,
        mut base_messages: Vec<Message>,
    ) -> anyhow::Result<Vec<Message>> {
        let llm_user_msg = match self.retrieve_context(prompt).await? {
            Some(ctx) => {
                Message::new(Role::User).with_contents([Part::text(format!("{ctx}\n\n{prompt}"))])
            }
            None => clean_user_msg,
        };
        base_messages.push(llm_user_msg);
        Ok(base_messages)
    }

    fn get_tool_descs(tools: &[Tool]) -> Vec<ToolDesc> {
        tools.iter().map(|v| v.get_description()).collect()
    }

    async fn handle_tool_calls(
        tools: &[Tool],
        tool_calls: Vec<Part>,
    ) -> anyhow::Result<Vec<MessageDelta>> {
        let mut tool_resps = Vec::new();
        for part in &tool_calls {
            let Some((id, name, args)) = part.as_function() else {
                continue;
            };
            let tool = tools
                .iter()
                .find(|v| v.get_description().name == name)
                .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", name))?
                .clone();
            let resp = tool.run(args.clone()).await?;
            let mut delta = MessageDelta::new()
                .with_role(Role::Tool)
                .with_contents([PartDelta::Value { value: resp }]);
            if let Some(id) = id {
                delta = delta.with_id(id);
            }
            tool_resps.push(delta);
        }
        Ok(tool_resps)
    }

    // ── Public APIs ───────────────────────────────────────────────────────────

    /// Stream complete turn outputs (one [`MessageOutput`] per model response or tool result).
    ///
    /// Handles the full agentic cycle: retrieval → inference → tool calls → repeat.
    /// The clean user message is stored in history; retrieved context is injected
    /// ephemerally into the LLM message only.
    pub fn stream_turns<'a>(
        &'a mut self,
        prompt: impl Into<String>,
        config: Option<AgentConfig>,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>> {
        let prompt = prompt.into();
        let (clean_user_msg, base_messages, tools, inference_config) =
            self.prepare_turn(prompt.clone(), config);

        Box::pin(async_stream::try_stream! {
            let mut messages = self.build_llm_messages(&prompt, clean_user_msg, base_messages).await?;

            let tool_descs = Self::get_tool_descs(&tools);
            loop {
                let model = self.lm.clone();
                let assistant_out = model
                    .infer(messages.clone(), tool_descs.clone(), inference_config.clone())
                    .await?;
                let assistant_msg = assistant_out.message.clone();
                messages.push(assistant_msg.clone());
                yield assistant_out;
                self.history.push(assistant_msg.clone());

                if let Some(tool_calls) = assistant_msg.tool_calls
                    && !tool_calls.is_empty()
                {
                    for delta in Self::handle_tool_calls(&tools, tool_calls).await? {
                        let message = delta.finish()?;
                        yield MessageOutput {
                            message: message.clone(),
                            finish_reason: FinishReason::Stop {},
                        };
                        self.history.push(message.clone());
                        messages.push(message);
                    }
                } else {
                    break;
                }
            }
        })
    }

    /// Stream per-token delta outputs ([`MessageDeltaOutput`]) for the given prompt.
    ///
    /// Same agentic cycle as [`stream_turns`] but yields incremental token deltas instead
    /// of complete messages — suitable for real-time display.
    pub fn stream_delta(
        &mut self,
        prompt: impl Into<String>,
        config: Option<AgentConfig>,
    ) -> BoxStream<'_, anyhow::Result<MessageDeltaOutput>> {
        let prompt = prompt.into();
        let (clean_user_msg, base_messages, tools, inference_config) =
            self.prepare_turn(prompt.clone(), config);

        Box::pin(async_stream::try_stream! {
            let mut messages = self.build_llm_messages(&prompt, clean_user_msg, base_messages).await?;

            let tool_descs = Self::get_tool_descs(&tools);
            loop {
                let mut assistant_msg_delta = MessageDelta::new().with_role(Role::Assistant);
                {
                    let model = self.lm.clone();
                    let mut strm = model.infer_delta(
                        messages.clone(),
                        tool_descs.clone(),
                        inference_config.clone(),
                    );
                    while let Some(out) = strm.next().await {
                        let out = out?;
                        assistant_msg_delta = assistant_msg_delta
                            .accumulate(out.clone().delta)
                            .map_err(|e| anyhow::anyhow!(e))?;
                        yield out;
                    }
                }
                let assistant_msg = assistant_msg_delta.finish()?;
                messages.push(assistant_msg.clone());
                self.history.push(assistant_msg.clone());

                if let Some(tool_calls) = assistant_msg.tool_calls
                    && !tool_calls.is_empty()
                {
                    for delta in Self::handle_tool_calls(&tools, tool_calls).await? {
                        let output = MessageDeltaOutput {
                            delta: delta.clone(),
                            finish_reason: Some(FinishReason::Stop {}),
                        };
                        yield output;
                        let tool_msg = delta.finish()?;
                        messages.push(tool_msg.clone());
                        self.history.push(tool_msg);
                    }
                } else {
                    break;
                }
            }
        })
    }

    /// Single-shot run: prompt → last assistant [`Message`].
    pub async fn run(
        &mut self,
        prompt: impl Into<String>,
        config: Option<AgentConfig>,
    ) -> anyhow::Result<Message> {
        let mut outputs: Vec<MessageOutput> = Vec::new();
        {
            let mut strm = self.stream_turns(prompt, config);
            while let Some(out) = strm.next().await {
                outputs.push(out?);
            }
        }
        outputs
            .into_iter()
            .filter(|o| o.message.role == Role::Assistant)
            .last()
            .map(|o| o.message)
            .ok_or_else(|| anyhow::anyhow!("No response from agent"))
    }

    /// Returns the conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::{multi_platform_async_trait, multi_platform_test, tool};
    use futures::StreamExt;

    use super::*;
    use crate::{
        knowledge::{Document, Knowledge},
        model::LangModel,
        to_value,
        value::Value,
    };

    struct MockKnowledge {
        docs: Vec<Document>,
    }

    #[multi_platform_async_trait]
    impl Knowledge for MockKnowledge {
        async fn retrieve(&self, _query: &str) -> anyhow::Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    #[tool(description = "Get current temperature for a city")]
    async fn temperature(_location: String, unit: String) -> anyhow::Result<Value> {
        match unit.as_str() {
            "Celsius" => Ok(to_value!("40")),
            "Fahrenheit" => Ok(to_value!("104")),
            _ => anyhow::bail!("unknown unit: {}", unit),
        }
    }

    #[multi_platform_test]
    async fn run_simple_chat() -> anyhow::Result<()> {
        let mut agent = Agent::builder()
            .model(LangModel::anthropic("claude-haiku-4-5-20251001").build()?)
            .build();

        let mut strm = Box::pin(agent.stream_delta("Hi, what's your name?", None));
        let mut accumulated = MessageDelta::new();
        while let Some(output) = strm.next().await {
            let output = output?;
            if let Some(_finish_reason) = output.finish_reason {
                let msg = accumulated.clone().finish()?;
                assert!(!msg.contents.is_empty());
            } else {
                accumulated = accumulated.accumulate(output.delta)?;
            }
        }
        Ok(())
    }

    #[multi_platform_test]
    async fn run_tool_call() -> anyhow::Result<()> {
        let mut agent = Agent::builder()
            .model(LangModel::anthropic("claude-haiku-4-5-20251001").build()?)
            .tool(temperature_tool())
            .build();

        let mut strm =
            Box::pin(agent.stream_turns("How hot is it currently in Dubai in Celsius?", None));
        let mut count = 0;
        while let Some(output) = strm.next().await {
            output?;
            count += 1;
        }
        assert!(count > 0);
        Ok(())
    }

    #[multi_platform_test]
    async fn run_with_builder() -> anyhow::Result<()> {
        let mut agent = Agent::builder()
            .model(LangModel::anthropic("claude-haiku-4-5-20251001").build()?)
            .system("You are a helpful assistant.")
            .build();

        let response = agent.run("Hi, what's your name?", None).await?;
        assert_eq!(response.role, Role::Assistant);
        assert!(!response.contents.is_empty());
        Ok(())
    }

    #[multi_platform_test]
    async fn run_multi_turn() -> anyhow::Result<()> {
        let mut agent = Agent::builder()
            .model(LangModel::anthropic("claude-haiku-4-5-20251001").build()?)
            .build();

        assert_eq!(agent.history().len(), 0);

        let response = agent.run("Hi, what's your name?", None).await?;
        assert_eq!(response.role, Role::Assistant);
        assert!(!response.contents.is_empty());
        assert!(agent.history().len() >= 2);

        let response2 = agent.run("What did I just ask you?", None).await?;
        assert_eq!(response2.role, Role::Assistant);
        assert!(agent.history().len() >= 4);

        agent.clear_history();
        assert_eq!(agent.history().len(), 0);
        Ok(())
    }

    #[multi_platform_test]
    async fn run_with_knowledge_source() -> anyhow::Result<()> {
        let knowledge = MockKnowledge {
            docs: vec![
                Document::new(
                    "Marie Curie was born on November 7, 1867, in Warsaw, in the Kingdom of Poland.",
                ),
                Document::new("Warsaw is the capital and largest city of Poland."),
                Document::new(
                    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.",
                ),
                Document::new("Poland is a country in Central Europe with Warsaw as its capital."),
                Document::new(
                    "Paris is the capital of France. Marie Curie spent much of her adult life in Paris.",
                ),
            ],
        };
        let mut agent = Agent::builder()
            .model(LangModel::anthropic("claude-haiku-4-5-20251001").build()?)
            .knowledge(knowledge)
            .build();

        let response = agent
            .run(
                "What is the capital of the country where Marie Curie was born?",
                None,
            )
            .await?;
        assert_eq!(response.role, Role::Assistant);
        assert!(!response.contents.is_empty());
        println!("{:?}", response.contents);

        // History must store the clean user message — no "[Retrieved Context]" block.
        let user_history = agent
            .history()
            .iter()
            .find(|m| m.role == Role::User)
            .expect("user message in history");
        let user_text = user_history
            .contents
            .iter()
            .find_map(|p| p.as_text())
            .unwrap_or_default();
        assert!(
            !user_text.contains("[Retrieved Context]"),
            "history should store clean user message, got: {user_text}"
        );
        Ok(())
    }
}
