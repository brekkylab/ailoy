use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
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
}

impl AgentBuilder {
    fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            system: None,
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

    pub fn build(self) -> Agent {
        Agent {
            lm: self.model.expect("AgentBuilder: model is required"),
            tools: self.tools,
            system: self.system,
            history: Vec::new(),
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
}

impl Agent {
    pub fn new(lm: LangModel, tools: impl IntoIterator<Item = Tool>) -> Self {
        Self {
            lm,
            tools: tools.into_iter().collect(),
            system: None,
            history: Vec::new(),
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

    pub fn stream_turns<'a>(
        &'a mut self,
        messages: Vec<Message>,
        config: Option<AgentConfig>,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        let tools = self.tools.clone();
        let inference_config = config.unwrap_or_default().inference.unwrap_or_default();
        let strm = async_stream::try_stream! {
            let tool_descs = Self::get_tool_descs(&tools);
            let mut messages = messages;
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
        };
        Box::pin(strm)
    }

    pub fn run_turns<'a>(
        &'a mut self,
        messages: Vec<Message>,
        config: Option<AgentConfig>,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>> {
        let tools = self.tools.clone();
        let inference_config = config.unwrap_or_default().inference.unwrap_or_default();
        let strm = async_stream::try_stream! {
            let tool_descs = Self::get_tool_descs(&tools);
            let mut messages = messages;
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
        };
        Box::pin(strm)
    }

    /// Single-shot run: string prompt -> final assistant Message.
    /// Runs the full agentic loop (tool calls resolved), accumulates history, and returns the last assistant message.
    pub async fn run(&mut self, prompt: impl Into<String>) -> anyhow::Result<Message> {
        let user_message = Message::new(Role::User).with_contents([Part::text(prompt.into())]);
        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            messages.push(Message::new(Role::System).with_contents([Part::text(system.clone())]));
        }
        messages.extend(self.history.clone());
        messages.push(user_message.clone());
        self.history.push(user_message);

        // Collect outputs (stream borrows &mut self, so collect first then push to history)
        let mut outputs: Vec<MessageOutput> = Vec::new();
        {
            let mut strm = self.run_turns(messages, None);
            while let Some(out) = strm.next().await {
                outputs.push(out?);
            }
        }
        // run_turns already pushed output messages to history; find last assistant message
        let last_assistant = outputs
            .into_iter()
            .filter(|o| o.message.role == Role::Assistant)
            .last()
            .map(|o| o.message);
        last_assistant.ok_or_else(|| anyhow::anyhow!("No response from agent"))
    }

    /// Streaming run: string prompt -> per-token delta stream. History is updated as the stream is consumed.
    pub fn stream(
        &mut self,
        prompt: impl Into<String>,
    ) -> BoxStream<'_, anyhow::Result<MessageDeltaOutput>> {
        let user_message = Message::new(Role::User).with_contents([Part::text(prompt.into())]);
        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            messages.push(Message::new(Role::System).with_contents([Part::text(system.clone())]));
        }
        messages.extend(self.history.clone());
        messages.push(user_message.clone());
        self.history.push(user_message);

        self.stream_turns(messages, None)
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
    use ailoy_macros::{multi_platform_test, tool};
    use futures::StreamExt;

    use super::*;
    use crate::{model::LangModel, to_value, value::Value};

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

        let mut strm = Box::pin(agent.stream_turns(
            vec![Message::new(Role::User).with_contents(vec![Part::text("Hi, what's your name?")])],
            None,
        ));
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

        let mut strm = Box::pin(agent.run_turns(
            vec![Message::new(Role::User).with_contents(vec![Part::text(
                "How hot is it currently in Dubai in Celsius?",
            )])],
            None,
        ));
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

        let response = agent.run("Hi, what's your name?").await?;
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

        let response = agent.run("Hi, what's your name?").await?;
        assert_eq!(response.role, Role::Assistant);
        assert!(!response.contents.is_empty());
        // history should have user + assistant messages
        assert!(agent.history().len() >= 2);

        let response2 = agent.run("What did I just ask you?").await?;
        assert_eq!(response2.role, Role::Assistant);
        // history should have grown
        assert!(agent.history().len() >= 4);

        agent.clear_history();
        assert_eq!(agent.history().len(), 0);
        Ok(())
    }
}
