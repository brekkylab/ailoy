use anyhow::Context;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    knowledge::{Knowledge, KnowledgeBehavior as _, KnowledgeConfig},
    model::{InferenceConfig, LangModel, LangModelInference as _},
    tool::{Tool, ToolBehavior as _},
    utils::{BoxFuture, BoxStream, log},
    value::{
        Delta, Document, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput,
        Part, PartDelta, Role, ToolDesc,
    },
};

/// Configuration for running the agent.
///
/// See `InferenceConfig` and `KnowledgeConfig` for more details.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct AgentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference: Option<InferenceConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge: Option<KnowledgeConfig>,
}

/// The Agent is the central orchestrator that connects the **language model**, **tools**, and **knowledge** components.
/// It manages the entire reasoning and action loop, coordinating how each subsystem contributes to the final response.
///
/// In essence, the Agent:
///
/// - Understands user input
/// - Interprets structured responses from the language model (such as tool calls)
/// - Executes tools as needed
/// - Retrieves and integrates contextual knowledge before or during inference
///
/// # Public APIs
/// - `run_delta`: Runs a user query and streams incremental deltas (partial outputs)
/// - `run`: Runs a user query and returns a complete message once all deltas are accumulated
///
/// ## Delta vs. Complete Message
/// A *delta* represents a partial piece of model output, such as a text fragment or intermediate reasoning step.
/// Deltas can be accumulated into a full message using the provided accumulation utilities.
/// This allows real-time streaming while preserving the ability to reconstruct the final structured result.
///
/// See `MessageDelta`.
///
/// # Components
/// - **Language Model**: Generates natural language and structured outputs. It interprets the conversation context and predicts the assistant’s next action.
/// - **Tool**: Represents external functions or APIs that the model can dynamically invoke. The `Agent` detects tool calls and automatically executes them during the reasoning loop.
/// - **Knowledge**: Provides retrieval-augmented reasoning by fetching relevant information from stored documents or databases. When available, the `Agent` enriches model input with these results before generating an answer.
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core"))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Agent {
    lm: LangModel,
    tools: Vec<Tool>,
    knowledge: Option<Knowledge>,
}

impl Agent {
    pub fn new(lm: LangModel, tools: impl IntoIterator<Item = Tool>) -> Self {
        Self {
            lm,
            tools: tools.into_iter().collect(),
            knowledge: None,
        }
    }

    pub fn with_knowledge(mut self, knowledge: Knowledge) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    pub fn get_lm(&self) -> LangModel {
        self.lm.clone()
    }

    pub fn get_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn knowledge(&self) -> Option<Knowledge> {
        self.knowledge.clone()
    }

    pub fn add_tools(&mut self, tools: Vec<Tool>) {
        for tool in tools.iter() {
            let tool_name = tool.get_description().name;

            // If the tool with same name already exists, skip adding the tool.
            if self
                .tools
                .iter()
                .find(|t| t.get_description().name == tool_name)
                .is_some()
            {
                log::warn(format!(
                    "Tool \"{}\" is already registered. Skip adding the tool.",
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
        self.tools.retain(|t| {
            let tool_name = t.get_description().name;
            // Remove the tool if its name belongs to `tool_names`
            !tool_names.contains(&tool_name)
        });
    }

    pub fn remove_tool(&mut self, tool_name: String) {
        self.remove_tools(vec![tool_name]);
    }

    pub fn set_knowledge(&mut self, knowledge: Knowledge) {
        self.knowledge = Some(knowledge);
    }

    pub fn remove_knowledge(&mut self) {
        self.knowledge = None;
    }

    fn get_docs<'a>(
        msgs: &Vec<Message>,
        knowledge: &Option<Knowledge>,
        knowledge_config: KnowledgeConfig,
    ) -> BoxFuture<'a, anyhow::Result<Vec<Document>>> {
        let last_msg = msgs.last().cloned();
        let knowledge = knowledge.clone();
        Box::pin(async move {
            if let Some(message) = last_msg
                && message.role == Role::User
                && let Some(knowledge) = knowledge.clone()
            {
                let query_str = message
                    .contents
                    .iter()
                    .filter(|p| p.is_text())
                    .map(|p| p.as_text().unwrap().to_owned())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                Ok(knowledge.retrieve(query_str, knowledge_config).await?)
            } else {
                Ok(vec![])
            }
        })
    }

    fn get_tool_descs(tools: &Vec<Tool>) -> Vec<ToolDesc> {
        tools
            .iter()
            .map(|v| v.get_description())
            .collect::<Vec<_>>()
    }

    async fn handle_tool_calls(
        tools: &Vec<Tool>,
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
                .unwrap()
                .clone();
            let resp = tool.run(args.clone()).await?;
            let mut delta =
                MessageDelta::new()
                    .with_role(Role::Tool)
                    .with_contents([PartDelta::Value {
                        value: resp.clone(),
                    }]);
            if let Some(id) = id {
                delta = delta.with_id(id);
            };
            tool_resps.push(delta);
        }
        Ok(tool_resps)
    }

    pub fn run_delta<'a>(
        &'a mut self,
        mut messages: Vec<Message>,
        config: Option<AgentConfig>,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        let knowledge = self.knowledge.clone();
        let tools = self.tools.clone();
        let AgentConfig {
            inference: inference_config,
            knowledge: knowledge_config,
        } = config.unwrap_or_default();
        let strm = async_stream::try_stream! {
            let docs = Self::get_docs(
                &messages,
                &knowledge,
                knowledge_config.unwrap_or_default()
            ).await?;
            let tool_descs = Self::get_tool_descs(&tools);
            loop {
                let mut assistant_msg_delta = MessageDelta::new().with_role(Role::Assistant);
                {
                    let mut model = self.lm.clone();
                    let mut strm = model.infer_delta(
                        messages.clone(),
                        tool_descs.clone(),
                        docs.clone(),
                        inference_config.clone().unwrap_or_default()
                    );
                    while let Some(out) = strm.next().await {
                        let out = out?;
                        assistant_msg_delta = assistant_msg_delta.accumulate(out.clone().delta).context("Aggregation failed")?;
                        if out.finish_reason.is_some() {
                            yield out;
                            break;
                        } else {
                            yield out;
                        }
                    }
                }
                let assistant_msg = assistant_msg_delta.clone().finish()?;
                messages.push(assistant_msg.clone());

                if let Some(tool_calls) = assistant_msg.tool_calls && !tool_calls.is_empty() {
                    for delta in Self::handle_tool_calls(&tools, tool_calls).await? {
                        let message_delta_output = MessageDeltaOutput { delta, finish_reason: Some(FinishReason::Stop{}) };
                        yield message_delta_output.clone();
                        messages.push(message_delta_output.delta.finish().unwrap());
                    }
                } else {
                    break;
                }
            }
        };
        Box::pin(strm)
    }

    pub fn run<'a>(
        &'a mut self,
        mut messages: Vec<Message>,
        config: Option<AgentConfig>,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>> {
        let knowledge = self.knowledge.clone();
        let tools = self.tools.clone();
        let AgentConfig {
            inference: inference_config,
            knowledge: knowledge_config,
        } = config.unwrap_or_default();
        let strm = async_stream::try_stream! {
            let docs = Self::get_docs(
                &messages,
                &knowledge,
                knowledge_config.unwrap_or_default()
            ).await?;
            let tool_descs = Self::get_tool_descs(&tools);
            loop {
                let mut model = self.lm.clone();
                let assistant_out = model.infer(
                    messages.clone(),
                    tool_descs.clone(),
                    docs.clone(),
                    inference_config.clone().unwrap_or_default()
                ).await?;
                let assistant_msg = assistant_out.message.clone();
                messages.push(assistant_msg.clone());
                yield assistant_out;

                if let Some(tool_calls) = assistant_msg.tool_calls && !tool_calls.is_empty() {
                    for delta in Self::handle_tool_calls(&tools, tool_calls).await? {
                        let message_output = MessageOutput { message: delta.finish()?, finish_reason: FinishReason::Stop{} };
                        yield message_output.clone();
                        messages.push(message_output.message);
                    }
                } else {
                    break;
                }
            }
        };
        Box::pin(strm)
    }
}

#[cfg(feature = "python")]
mod py {
    use std::sync::Arc;

    use futures::lock::Mutex;
    use pyo3::pymethods;
    use pyo3_stub_gen_derive::gen_stub_pymethods;
    use tokio::sync::mpsc;

    use super::*;
    use crate::value::{
        Messages,
        py::{MessageDeltaOutputIterator, MessageDeltaOutputSyncIterator, MessageOutputIterator},
    };

    fn spawn_delta<'a>(
        mut agent: Agent,
        messages: Vec<Message>,
        config: Option<AgentConfig>,
    ) -> anyhow::Result<(
        tokio::runtime::Runtime,
        mpsc::UnboundedReceiver<anyhow::Result<MessageDeltaOutput>>,
    )> {
        let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<MessageDeltaOutput>>();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            let mut stream = agent.run_delta(messages, config).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break; // Exit if consumer vanished
                }
            }
        });
        Ok((rt, rx))
    }

    fn spawn<'a>(
        mut agent: Agent,
        messages: Vec<Message>,
        config: Option<AgentConfig>,
    ) -> anyhow::Result<(
        tokio::runtime::Runtime,
        mpsc::UnboundedReceiver<anyhow::Result<MessageOutput>>,
    )> {
        let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<MessageOutput>>();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            let mut stream = agent.run(messages, config).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break; // Exit if consumer vanished
                }
            }
        });
        Ok((rt, rx))
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl AgentConfig {
        #[new]
        #[pyo3(signature = (inference=None, knowledge=None))]
        fn __new__(inference: Option<InferenceConfig>, knowledge: Option<KnowledgeConfig>) -> Self {
            Self {
                inference,
                knowledge,
            }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl Agent {
        #[new]
        #[pyo3(signature = (lm, tools = None))]
        fn __new__(lm: LangModel, tools: Option<Vec<Tool>>) -> Self {
            Agent::new(lm, tools.unwrap_or_default())
        }

        pub fn __repr__(&self) -> String {
            format!(
                "Agent(lm={}, tools=[{} items])",
                self.get_lm().__repr__(),
                self.tools.len()
            )
        }

        #[getter]
        fn lm(&self) -> LangModel {
            self.get_lm()
        }

        #[getter]
        fn tools(&self) -> Vec<Tool> {
            self.get_tools()
        }

        #[pyo3(name="add_tools", signature = (tools))]
        fn add_tools_py(&mut self, tools: Vec<Tool>) {
            self.add_tools(tools);
        }

        #[pyo3(name="add_tool", signature = (tool))]
        fn add_tool_py(&mut self, tool: Tool) {
            self.add_tool(tool);
        }

        #[pyo3(name="remove_tools", signature = (tool_names))]
        fn remove_tools_py(&mut self, tool_names: Vec<String>) {
            self.remove_tools(tool_names);
        }

        #[pyo3(name="remove_tool", signature = (tool_name))]
        fn remove_tool_py(&mut self, tool_name: String) {
            self.remove_tool(tool_name);
        }

        #[pyo3(name="run_delta", signature = (messages, config=None))]
        fn run_delta_py(
            &mut self,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> anyhow::Result<MessageDeltaOutputIterator> {
            let (_rt, rx) = spawn_delta(self.clone(), messages.into(), config)?;
            Ok(MessageDeltaOutputIterator {
                _rt,
                rx: Arc::new(Mutex::new(rx)),
            })
        }

        #[pyo3(name="run_delta_sync", signature = (messages, config=None))]
        fn run_delta_sync_py(
            &mut self,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> anyhow::Result<MessageDeltaOutputSyncIterator> {
            let (_rt, rx) = spawn_delta(self.clone(), messages.into(), config)?;
            Ok(MessageDeltaOutputSyncIterator { _rt, rx })
        }

        #[pyo3(name="run", signature = (messages, config=None))]
        fn run_py(
            &mut self,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> anyhow::Result<MessageOutputIterator> {
            let (_rt, rx) = spawn(self.clone(), messages.into(), config)?;
            Ok(MessageOutputIterator {
                _rt,
                rx: Arc::new(Mutex::new(rx)),
            })
        }

        #[pyo3(name="run_sync", signature = (messages, config=None))]
        fn run_sync_py(
            &mut self,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> anyhow::Result<MessageDeltaOutputSyncIterator> {
            let (_rt, rx) = spawn_delta(self.clone(), messages.into(), config)?;
            Ok(MessageDeltaOutputSyncIterator { _rt, rx })
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use std::sync::Arc;

    use futures::lock::Mutex;
    use napi::bindgen_prelude::*;
    use napi_derive::napi;
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        ffi::node::common::get_or_create_runtime,
        value::{
            Messages,
            node::{MessageDeltaOutputIterator, MessageOutputIterator},
        },
    };

    #[napi]
    impl Agent {
        #[napi(constructor)]
        pub fn new_js(lm: &LangModel, tools: Option<Vec<&Tool>>) -> Self {
            Self::new(
                lm.clone(),
                tools.unwrap_or(vec![]).iter().map(|&t| t.clone()),
            )
        }

        #[napi(js_name = "addTool")]
        pub unsafe fn add_tool_js(&mut self, tool: &Tool) {
            self.add_tool(tool.clone())
        }

        #[napi(js_name = "addTools")]
        pub unsafe fn add_tools_js(&mut self, tools: Vec<&Tool>) {
            self.add_tools(tools.iter().map(|&t| t.clone()).collect())
        }

        #[napi(js_name = "removeTool")]
        pub unsafe fn remove_tool_js(&mut self, tool_name: String) {
            self.remove_tool(tool_name)
        }

        #[napi(js_name = "removeTools")]
        pub unsafe fn remove_tools_js(&mut self, tool_names: Vec<String>) {
            self.remove_tools(tool_names)
        }

        #[napi(js_name = "setKnowledge")]
        pub unsafe fn set_knowledge_js(&mut self, knowledge: &Knowledge) {
            self.set_knowledge(knowledge.clone())
        }

        #[napi(js_name = "removeKnowledge")]
        pub unsafe fn remove_knowledge_js(&mut self) {
            self.remove_knowledge()
        }

        #[napi(js_name = "runDelta", ts_return_type = "MessageDeltaOutputIterator")]
        pub fn run_delta_js<'a>(
            &'a mut self,
            env: Env,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> napi::Result<Object<'a>> {
            let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<MessageDeltaOutput>>();
            let rt = get_or_create_runtime();
            let mut agent = self.clone();

            rt.spawn(async move {
                let mut stream = agent.run_delta(messages.into(), config).boxed();

                while let Some(item) = stream.next().await {
                    if tx
                        .send(item.map_err(|e| anyhow::anyhow!(e.to_string())))
                        .is_err()
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            });

            let it = MessageDeltaOutputIterator {
                rx: Arc::new(Mutex::new(rx)),
            };
            it.to_async_iterator(env)
        }

        #[napi(js_name = "run", ts_return_type = "MessageOutputIterator")]
        pub fn run_js<'a>(
            &'a mut self,
            env: Env,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> napi::Result<Object<'a>> {
            let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<MessageOutput>>();
            let rt = get_or_create_runtime();
            let mut agent = self.clone();

            rt.spawn(async move {
                let mut stream = agent.run(messages.into(), config).boxed();

                while let Some(item) = stream.next().await {
                    if tx
                        .send(item.map_err(|e| anyhow::anyhow!(e.to_string())))
                        .is_err()
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            });

            let it = MessageOutputIterator {
                rx: Arc::new(Mutex::new(rx)),
            };
            it.to_async_iterator(env)
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::{ffi::web::stream_to_async_iterable, value::Messages};

    #[wasm_bindgen]
    impl Agent {
        /// Construct a new Agent instance with provided `LangModel` and `Tool`s.
        ///
        /// Note that the ownership of `tools` is moved to the agent, which means you can't directly accessible to `tools` after the agent is initialized.
        /// If you still want to reuse the `tools`, try to use `addTool()` multiple times instead.
        #[wasm_bindgen(constructor)]
        pub fn new_js(lm: &LangModel, tools: Option<Vec<Tool>>) -> Self {
            Self::new(lm.clone(), tools.unwrap_or(vec![]))
        }

        #[wasm_bindgen(js_name = "addTool")]
        pub fn add_tool_js(&mut self, tool: &Tool) {
            self.add_tool(tool.clone())
        }

        #[wasm_bindgen(js_name = "removeTool")]
        pub fn remove_tool_js(&mut self, #[wasm_bindgen(js_name = "toolName")] tool_name: String) {
            self.remove_tool(tool_name)
        }

        #[wasm_bindgen(js_name = "setKnowledge")]
        pub fn set_knowledge_js(&mut self, knowledge: &Knowledge) {
            self.set_knowledge(knowledge.clone())
        }

        #[wasm_bindgen(js_name = "removeKnowledge")]
        pub fn remove_knowledge_js(&mut self) {
            self.remove_knowledge()
        }

        #[wasm_bindgen(
            js_name = "runDelta",
            unchecked_return_type = "AsyncIterable<MessageDeltaOutput>"
        )]
        pub fn run_delta_js(
            &self,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> Result<JsValue, js_sys::Error> {
            let mut agent = self.clone();
            let messages = messages.try_into()?;
            let stream = async_stream::stream! {
                let mut inner_stream = agent.run_delta(messages, config);
                while let Some(item) = inner_stream.next().await {
                    yield item;
                }
            };
            let js_stream = Box::pin(stream.map(|response| {
                response
                    .map(|resp| resp.into())
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }));

            Ok(stream_to_async_iterable(js_stream).into())
        }

        #[wasm_bindgen(
            js_name = "run",
            unchecked_return_type = "AsyncIterable<MessageOutput>"
        )]
        pub fn run_js(
            &self,
            messages: Messages,
            config: Option<AgentConfig>,
        ) -> Result<JsValue, js_sys::Error> {
            let mut agent = self.clone();
            let messages = messages.try_into()?;
            let stream = async_stream::stream! {
                let mut inner_stream = agent.run(messages, config);
                while let Some(item) = inner_stream.next().await {
                    yield item;
                }
            };
            let js_stream = Box::pin(stream.map(|response| {
                response
                    .map(|resp| resp.into())
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }));

            Ok(stream_to_async_iterable(js_stream).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;

    #[multi_platform_test]
    async fn run_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::model::LangModel;

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, Vec::new());

        let mut strm = Box::pin(agent.run_delta(
            vec![Message::new(Role::User).with_contents(vec![Part::text("Hi, what's your name?")])],
            None,
        ));
        let mut accumulated = MessageDelta::new();
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            accumulated = accumulated.accumulate(output.delta).unwrap();
        }
        println!("message: {:?}", accumulated.finish().unwrap());
    }

    #[multi_platform_test]
    async fn run_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::{
            model::LangModel,
            to_value,
            tool::ToolFunc,
            value::{ToolDesc, Value},
        };

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();

        let tool_desc = ToolDesc::new(
            "temperature".to_owned(),
            Some("Get current temperature".to_owned()),
            to_value!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["Celsius", "Fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            }),
            Some(to_value!({
                "type": "number",
                "description": "Null if the given city name is unavailable.",
                "nullable": true,
            })),
        );
        let tools = vec![Tool::new_function(
            tool_desc,
            std::sync::Arc::<Box<ToolFunc>>::new(Box::new(move |args: Value| {
                Box::pin(async move {
                    use anyhow::bail;

                    let unit = args
                        .as_object()
                        .unwrap()
                        .get("unit")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    match unit {
                        "Celsius" => Ok(to_value!("40")),
                        "Fahrenheit" => Ok(to_value!("104")),
                        _ => bail!(""),
                    }
                })
            })),
        )];

        let mut agent = Agent::new(model, tools);

        let mut strm = Box::pin(agent.run_delta(
            vec![
                    Message::new(Role::User)
                        .with_contents(vec![Part::text("How much hot currently in Dubai?")]),
                ],
            None,
        ));
        let mut accumulated = MessageDelta::new();
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            accumulated = accumulated.accumulate(output.delta).unwrap();
        }
        println!("message: {:?}", accumulated.finish().unwrap());
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_mcp_stdio_tool_call() {
        use futures::StreamExt;
        use rmcp::transport::ConfigureCommandExt;

        use super::*;
        use crate::{model::LangModel, tool::MCPClient};

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, Vec::new());

        let command = tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        });
        let mcp_client = MCPClient::from_stdio(command).await.unwrap();
        let mcp_tools = mcp_client
            .get_tools()
            .iter()
            .map(|tool| Tool::new_mcp(tool.clone()))
            .collect();
        agent.add_tools(mcp_tools);

        let agent_tools = agent.get_tools();
        assert_eq!(agent_tools.len(), 2);
        assert_eq!(agent_tools[0].get_description().name, "get_current_time");
        assert_eq!(agent_tools[1].get_description().name, "convert_time");

        let mut strm = Box::pin(agent.run_delta(
            vec![Message::new(Role::User).with_contents(vec![Part::text(
                "What time is it now in America/New_York timezone?",
            )])],
            None,
        ));
        let mut accumulated = MessageDelta::new();
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            accumulated = accumulated.accumulate(output.delta).unwrap();
        }
        println!("message: {:?}", accumulated.finish().unwrap());
    }
}
