use std::sync::Arc;

use anyhow::Context;
use futures::{StreamExt, lock::Mutex};
use serde::{Deserialize, Serialize};

use crate::{
    knowledge::{Knowledge, KnowledgeBehavior as _, KnowledgeConfig},
    model::{InferenceConfig, LangModel, LangModelInference as _},
    tool::{Tool, ToolBehavior as _},
    utils::{BoxStream, log},
    value::{Delta, FinishReason, Message, MessageDelta, Part, PartDelta, Role},
};

#[derive(Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Agent {
    lm: LangModel,
    tools: Vec<Tool>,
    messages: Arc<Mutex<Vec<Message>>>,
    knowledge: Option<Knowledge>,
}

/// The yielded value from agent.run().
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct AgentResponse {
    /// The message delta per iteration.
    pub delta: MessageDelta,
    /// Optional finish reason. If this is Some, the message aggregation is finalized and stored in `aggregated`.
    pub finish_reason: Option<FinishReason>,
    /// Optional aggregated message.
    pub aggregated: Option<Message>,
}

impl Agent {
    pub fn new(lm: LangModel, tools: impl IntoIterator<Item = Tool>) -> Self {
        Self {
            lm,
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
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

    pub async fn clear_messages(&mut self) -> anyhow::Result<()> {
        self.messages.lock().await.clear();
        Ok(())
    }

    pub async fn add_tools(&mut self, tools: Vec<Tool>) -> anyhow::Result<()> {
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

        Ok(())
    }

    pub async fn add_tool(&mut self, tool: Tool) -> anyhow::Result<()> {
        self.add_tools(vec![tool]).await
    }

    pub async fn remove_tools(&mut self, tool_names: Vec<String>) -> anyhow::Result<()> {
        self.tools.retain(|t| {
            let tool_name = t.get_description().name;
            // Remove the tool if its name belongs to `tool_names`
            !tool_names.contains(&tool_name)
        });
        Ok(())
    }

    pub async fn remove_tool(&mut self, tool_name: String) -> anyhow::Result<()> {
        self.remove_tools(vec![tool_name]).await
    }

    pub fn set_knowledge(&mut self, knowledge: Knowledge) {
        self.knowledge = Some(knowledge);
    }

    pub fn remove_knowledge(&mut self) {
        self.knowledge = None;
    }

    pub fn run<'a>(
        &'a mut self,
        contents: Vec<Part>,
    ) -> BoxStream<'a, anyhow::Result<AgentResponse>> {
        // Messages used for the current run round
        let mut messages: Vec<Message> = vec![];
        // For storing message history except system message
        let messages_history = self.messages.clone();

        let tools = self.tools.clone();
        let strm = async_stream::try_stream! {
            // Get documents
            let docs = if let Some(knowledge) = self.knowledge.clone() {
                let query_str = contents.iter().filter(|p| p.is_text()).map(|p| p.as_text().unwrap().to_owned()).collect::<Vec<_>>().join("\n\n");
                knowledge.retrieve(query_str, KnowledgeConfig::default()).await?
            } else {
                vec![]
            };

            // Add message histories to messages
            for msg in messages_history.lock().await.iter() {
                messages.push(msg.clone());
            }

            let user_message = Message::new(Role::User).with_contents(contents);
            // Add user message to messages
            messages.push(user_message.clone());
            // Add user message to histories
            messages_history.lock().await.push(user_message);

            let tool_descs = self
                .tools
                .iter()
                .map(|v| v.get_description())
                .collect::<Vec<_>>();
            loop {
                let mut assistant_msg_delta = MessageDelta::new().with_role(Role::Assistant);
                let mut assistant_msg: Option<Message> = None;
                {
                    let mut model = self.lm.clone();
                    let mut strm = model.infer(messages.clone(), tool_descs.clone(), docs.clone(), InferenceConfig::default());
                    while let Some(out) = strm.next().await {
                        let out = out?;
                        assistant_msg_delta = assistant_msg_delta.aggregate(out.clone().delta).context("Aggregation failed")?;

                        // Message aggregation is finalized if finish_reason does exist
                        if out.finish_reason.is_some() {
                            assistant_msg = Some(assistant_msg_delta.clone().finish()?);
                        }
                        yield AgentResponse {
                            delta: out.delta.clone(),
                            finish_reason: out.finish_reason.clone(),
                            aggregated: assistant_msg.clone(),
                        };
                    }
                }
                let assistant_msg = assistant_msg.unwrap();
                // Add assistant message to messages
                messages.push(assistant_msg.clone());
                // Add assistant message to histories
                messages_history.lock().await.push(assistant_msg.clone());

                // Handling tool calls if exist
                if let Some(tool_calls) = assistant_msg.tool_calls && !tool_calls.is_empty() {
                    for part in &tool_calls {
                        let Some((id, name, args)) = part.as_function() else { continue; };
                        let tool = tools.iter().find(|v| v.get_description().name == name).unwrap().clone();
                        let resp = tool.run(args.clone()).await?;
                        let mut tool_msg_delta = MessageDelta::new().with_role(Role::Tool).with_contents([PartDelta::Value{value: resp.clone()}]);
                        let mut tool_msg = Message::new(Role::Tool).with_contents([Part::Value{value: resp.clone()}]);
                        if let Some(id) = id {
                            tool_msg_delta = tool_msg_delta.with_id(id);
                            tool_msg = tool_msg.with_id(id);
                        }
                        yield AgentResponse {
                            delta: tool_msg_delta,
                            finish_reason: Some(FinishReason::Stop{}),
                            aggregated: Some(tool_msg.clone()),
                        };

                        // Add tool message to messages
                        messages.push(tool_msg.clone());
                        // Add tool message to histories
                        messages_history.lock().await.push(tool_msg);
                    }
                } else {
                    break;
                }
            }
        };
        Box::pin(strm)
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::ffi::web::stream_to_async_iterable;

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
        pub async fn add_tool_js(&mut self, tool: &Tool) -> Result<(), js_sys::Error> {
            self.add_tool(tool.clone())
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "removeTool")]
        pub async fn remove_tool_js(
            &mut self,
            #[wasm_bindgen(js_name = "toolName")] tool_name: String,
        ) -> Result<(), js_sys::Error> {
            self.remove_tool(tool_name)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "setKnowledge")]
        pub fn set_knowledge_js(&mut self, knowledge: &Knowledge) {
            self.set_knowledge(knowledge.clone());
        }

        #[wasm_bindgen(js_name = "removeKnowledge")]
        pub fn remove_knowledge_js(&mut self) {
            self.remove_knowledge();
        }

        #[wasm_bindgen(
            js_name = "run",
            unchecked_return_type = "AsyncIterable<AgentResponse>"
        )]
        pub fn run_js(&self, contents: Vec<Part>) -> JsValue {
            let mut agent = self.clone();
            let stream = async_stream::stream! {
                let mut inner_stream = agent.run(contents);
                while let Some(item) = inner_stream.next().await {
                    yield item;
                }
            };
            let js_stream = Box::pin(stream.map(|response| {
                response
                    .map(|resp| resp.into())
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }));

            stream_to_async_iterable(js_stream).into()
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_simple_chat() -> anyhow::Result<()> {
        use futures::StreamExt;

        use super::*;
        use crate::model::LangModel;

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, Vec::new());

        let mut strm = Box::pin(agent.run(vec![Part::text("Hi what's your name?")]));
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            if output.aggregated.is_some() {
                println!("message: {:?}", output.aggregated.unwrap());
            }
        }

        Ok(())
    }

    // #[cfg(any(target_family = "unix", target_family = "windows"))]
    // #[tokio::test]
    // async fn run_tool_call() {
    //     use futures::StreamExt;
    //     use serde_json::json;

    //     use super::*;
    //     use crate::{model::LocalLanguageModel, tool::BuiltinTool, value::ToolDesc};

    //     let cache = crate::cache::Cache::new();
    //     let key = "Qwen/Qwen3-0.6B";
    //     let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
    //     let mut model: Option<LocalLanguageModel> = None;
    //     while let Some(progress) = model_strm.next().await {
    //         let mut progress = progress.unwrap();
    //         println!("{} / {}", progress.current_task, progress.total_task);
    //         if progress.current_task == progress.total_task {
    //             model = progress.result.take();
    //         }
    //     }
    //     let model = model.unwrap();
    //     let tool_desc = ToolDesc::new(
    //         "temperature".into(),
    //         "Get current temperature".into(),
    //         json!({
    //             "type": "object",
    //             "properties": {
    //                 "location": {
    //                     "type": "string",
    //                     "description": "The city name"
    //                 },
    //                 "unit": {
    //                     "type": "string",
    //                     "enum": ["Celsius", "Fahrenheit"]
    //                 }
    //             },
    //             "required": ["location", "unit"]
    //         }),
    //         Some(json!({
    //             "type": "number",
    //             "description": "Null if the given city name is unavailable.",
    //             "nullable": true,
    //         })),
    //     )
    //     .unwrap();
    //     let tools = vec![Arc::new(BuiltinTool::new(
    //         tool_desc,
    //         Arc::new(|args| {
    //             if args
    //                 .as_object()
    //                 .unwrap()
    //                 .get("unit")
    //                 .unwrap()
    //                 .as_str()
    //                 .unwrap()
    //                 == "Celsius"
    //             {
    //                 Part::Text("40".to_owned())
    //             } else {
    //                 Part::Text("104".to_owned())
    //             }
    //         }),
    //     )) as Tool];
    //     let mut agent = Agent::new(model, tools);

    //     let mut agg = MessageAggregator::new();
    //     let mut strm =
    //         Box::pin(agent.run(vec![Part::Text("How much hot currently in Dubai?".into())]));
    //     while let Some(delta_opt) = strm.next().await {
    //         let delta = delta_opt.unwrap();
    //         if let Some(msg) = agg.update(delta) {
    //             println!("{:?}", msg);
    //         }
    //     }
    // }

    // #[cfg(any(target_family = "unix", target_family = "windows"))]
    // #[tokio::test]
    // async fn run_mcp_stdio_tool_call() -> anyhow::Result<()> {
    //     use futures::StreamExt;

    //     use super::*;
    //     use crate::{model::LocalLanguageModel, tool::MCPTransport};

    //     let cache = crate::cache::Cache::new();
    //     let key = "Qwen/Qwen3-0.6B";
    //     let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
    //     let mut model: Option<LocalLanguageModel> = None;
    //     while let Some(progress) = model_strm.next().await {
    //         let mut progress = progress.unwrap();
    //         println!("{} / {}", progress.current_task, progress.total_task);
    //         if progress.current_task == progress.total_task {
    //             model = progress.result.take();
    //         }
    //     }
    //     let model = model.unwrap();

    //     let mut agent = Agent::new(model, vec![]);
    //     let transport = MCPTransport::Stdio {
    //         command: "uvx".into(),
    //         args: vec!["mcp-server-time".into()],
    //     };
    //     agent
    //         .add_tools(transport.get_tools("time").await.unwrap())
    //         .await
    //         .unwrap();

    //     let agent_tools = agent.get_tools();
    //     assert_eq!(agent_tools.len(), 2);
    //     assert_eq!(
    //         agent_tools[0].get_description().name,
    //         "time--get_current_time"
    //     );
    //     assert_eq!(agent_tools[1].get_description().name, "time--convert_time");

    //     let mut agg = MessageAggregator::new();
    //     let mut strm = Box::pin(agent.run(vec![Part::Text(
    //         "What time is it now in America/New_York timezone?".into(),
    //     )]));
    //     while let Some(delta_opt) = strm.next().await {
    //         let delta = delta_opt.unwrap();
    //         if let Some(msg) = agg.update(delta) {
    //             println!("{:?}", msg);
    //         }
    //     }

    //     Ok(())
    // }
}
