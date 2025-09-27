use std::sync::Arc;

use futures::{Stream, StreamExt, lock::Mutex};

use crate::{
    model::{ArcMutexLanguageModel, InferenceConfig, LanguageModel},
    tool::Tool,
    utils::log,
    value::{Delta, FinishReason, Message, MessageDelta, MessageOutput, Part, PartDelta, Role},
};

// #[derive(Clone)]
pub struct Agent {
    lm: ArcMutexLanguageModel,
    tools: Vec<Arc<dyn Tool>>,
    messages: Arc<Mutex<Vec<Message>>>,
}

impl Agent {
    pub fn new(
        lm: impl LanguageModel + 'static,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Self {
        Self {
            lm: ArcMutexLanguageModel::new(lm),
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_from_arc(
        lm: ArcMutexLanguageModel,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Self {
        Self {
            lm,
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_lm(&self) -> ArcMutexLanguageModel {
        self.lm.clone()
    }

    pub fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    pub async fn add_tools(&mut self, tools: Vec<Arc<dyn Tool>>) -> anyhow::Result<()> {
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

    pub async fn add_tool(&mut self, tool: Arc<dyn Tool>) -> anyhow::Result<()> {
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

    pub fn run<'a>(
        &'a mut self,
        contents: Vec<Part>,
    ) -> impl Stream<Item = Result<MessageOutput, String>> {
        let tools = self.tools.clone();
        let msgs = self.messages.clone();
        let user_message = Message::new(Role::User).with_contents(contents);
        async_stream::try_stream! {
            msgs.lock().await.push(user_message);
            let td = self
                .tools
                .iter()
                .map(|v| v.get_description())
                .collect::<Vec<_>>();
            loop {
                let mut assistant_msg = MessageDelta::new().with_role(Role::Assistant);
                {
                    let mut model = self.lm.model.lock().await;
                    let mut strm = model.run(msgs.lock().await.clone(), td.clone(), InferenceConfig::default());
                    while let Some(output_opt) = strm.next().await {
                        let output = output_opt?;
                        yield output.clone();
                        assistant_msg = assistant_msg.aggregate(output.delta).map_err(|_| String::from("Aggregation failed"))?;
                    }
                }
                let assistant_msg = assistant_msg.finish()?;

                msgs.lock().await.push(assistant_msg.clone());
                if !assistant_msg.tool_calls.is_empty() {
                    for part in &assistant_msg.tool_calls {
                        let Some((id, name, args)) = part.as_function() else { continue };
                        let tool = tools.iter().find(|v| v.get_description().name == name).unwrap().clone();
                        let resp = tool.run(args.to_owned()).await?;
                        let mut msg = Message::new(Role::Tool).with_contents([Part::Value { value: resp.clone() }]);
                        let mut delta = MessageDelta::new().with_role(Role::Tool).with_contents([PartDelta::Value { value: resp }]);
                        if let Some(id) = id {
                            msg = msg.with_id(id);
                            delta = delta.with_id(id);
                        }
                        yield MessageOutput{delta, finish_reason: Some(FinishReason::Stop())};
                        msgs.lock().await.push(msg);
                    }
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // #[cfg(any(target_family = "unix", target_family = "windows"))]
    // #[tokio::test]
    // async fn run_simple_chat() {
    //     use futures::StreamExt;

    //     use super::*;
    //     use crate::model::LocalLanguageModel;

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
    //     let mut agent = Agent::new(model, Vec::new());

    //     let mut agg = MessageAggregator::new();
    //     let mut strm = Box::pin(agent.run(vec![Part::Text("Hi what's your name?".into())]));
    //     while let Some(delta_opt) = strm.next().await {
    //         let delta = delta_opt.unwrap();
    //         if let Some(msg) = agg.update(delta) {
    //             println!("{:?}", msg);
    //         }
    //     }
    // }

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
    //     )) as Arc<dyn Tool>];
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
