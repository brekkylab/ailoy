use std::sync::Arc;

use futures::{Stream, StreamExt};
use tokio::sync::Mutex;

use crate::{
    model::LanguageModel,
    tool::{MCPTransport, Tool},
    value::{
        FinishReason, Message, MessageAggregator, MessageOutput, Part, Role, ToolCall, ToolCallArg,
    },
};

#[derive(Clone)]
pub struct Agent {
    lm: Box<dyn LanguageModel>,
    tools: Vec<Arc<dyn Tool>>,
    messages: Arc<Mutex<Vec<Message>>>,
}

impl Agent {
    pub fn new(
        lm: impl LanguageModel + 'static,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Self {
        Self {
            lm: Box::new(lm),
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_lm(&self) -> &Box<dyn LanguageModel> {
        &self.lm
    }

    pub fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    pub async fn add_mcp_tools(
        &mut self,
        client_name: &str,
        transport: MCPTransport,
        tools_to_add: Vec<String>,
    ) -> anyhow::Result<()> {
        let mut tools = transport.get_tools(client_name).await?;

        // If tools_to_add is not empty, filter out the tools not in the whitelist.
        if !tools_to_add.is_empty() {
            tools.retain(|t| {
                let tool_desc = t.get_description();
                let tool_name = tool_desc.name.split("--").last().unwrap();
                tools_to_add.contains(&tool_name.to_string())
            })
        }

        for tool in tools.iter() {
            let tool_name = tool.get_description().name;

            // If the tool with same name already exists, skip adding the tool.
            if self
                .tools
                .iter()
                .find(|t| t.get_description().name == tool_name)
                .is_some()
            {
                println!(
                    "MCP tool \"{}\" is already registered. Skip adding the tool.",
                    tool_name
                );
                continue;
            }

            self.tools.push(tool.clone());
        }

        Ok(())
    }

    pub async fn remove_mcp_tools(&mut self, client_name: &str) -> anyhow::Result<()> {
        // Remove MCP tools
        self.tools.retain(|t| {
            let tool_name = t.get_description().name;
            // Remove the MCP tool if its description name is prefixed with the provided client name.
            !tool_name.starts_with(format!("{}--", client_name).as_str())
        });

        Ok(())
    }

    pub fn run<'a>(
        &'a mut self,
        user_message: impl Into<String>,
    ) -> impl Stream<Item = Result<MessageOutput, String>> {
        let tools = self.tools.clone();
        let msgs = self.messages.clone();
        let user_message =
            Message::with_role(Role::User).with_contents(vec![Part::Text(user_message.into())]);
        async_stream::try_stream! {
            msgs.lock().await.push(user_message);
            let td = self
                .tools
                .iter()
                .map(|v| v.get_description())
                .collect::<Vec<_>>();
            loop {
                let mut assistant_msg = Message::with_role(Role::Assistant);
                {
                    let mut strm = self.lm.run(msgs.lock().await.clone(), td.clone());
                    let mut aggregator = MessageAggregator::new();
                    while let Some(delta) = strm.next().await {
                        let delta = delta?;
                        yield delta.clone();
                        if let Some(msg) = aggregator.update(delta) {
                            assistant_msg = msg;
                        }
                    }
                }

                msgs.lock().await.push(assistant_msg.clone());
                if !assistant_msg.tool_calls.is_empty() {
                    for part in &assistant_msg.tool_calls {
                        let (tc, tool_call_id) = match part {
                            Part::FunctionString(s) => match ToolCall::try_from_string(s.clone()){
                                Ok(tc) => (tc, None),
                                Err(_) => { continue; },
                            },
                            Part::Function{id, name, arguments} => {
                                let Ok(arguments) = ToolCallArg::try_from_string(arguments) else {
                                    continue;
                                };
                                (ToolCall{name: name.to_owned(), arguments}, Some(id.clone()))
                            },
                            _ => {continue;},
                        };
                        let tool = tools.iter().find(|v| v.get_description().get_name() == tc.name).unwrap().clone();
                        let resp = tool.run(tc.arguments).await?;
                        let mut delta = Message::with_role(Role::Tool).with_contents(resp.clone());
                        if let Some(tool_call_id) = tool_call_id {
                            delta = delta.with_tool_call_id(tool_call_id);
                        }
                        yield MessageOutput::new().with_delta(delta.clone()).with_finish_reason(FinishReason::Stop);
                        msgs.lock().await.push(delta);
                    }
                }
                // msgs.lock().await.push(assistant_msg.clone());
                // if !assistant_msg.tool_calls.is_empty() {
                //     for part in assistant_msg.tool_calls {
                //         let tool_call_id = if let Part::Function{id, ..} = part.clone() {
                //             Some(id.clone())
                //         } else {
                //             None
                //         };
                //         let tc: ToolCall = part.try_into().unwrap();
                //         let tool = tools.iter().find(|v| v.get_description().name == tc.name).unwrap().clone();
                //         let parts = tool.run(tc.arguments).await?;
                //         for part in parts.into_iter() {
                //             let mut tool_msg = Message::with_role(Role::Tool).with_contents([part.clone()]);
                //             if let Some(tool_call_id) = tool_call_id.clone() {
                //                 tool_msg = tool_msg.with_tool_call_id(tool_call_id);
                //             }
                //             yield MessageOutput{ delta: tool_msg.clone(), finish_reason: Some(FinishReason::Stop)};
                //             self.messages.lock().await.push(tool_msg);
                //         }
                //     }
                // }
                 else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::model::LocalLanguageModel;

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!("{} / {}", progress.current_task, progress.total_task);
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = model.unwrap();
        let mut agent = Agent::new(model, Vec::new());

        let mut agg = MessageAggregator::new();
        let mut strm = Box::pin(agent.run("Hi what's your name?"));
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::{
            model::LocalLanguageModel,
            tool::BuiltinTool,
            value::{ToolDesc, ToolDescArg},
        };

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!("{} / {}", progress.current_task, progress.total_task);
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = model.unwrap();
        let tools = vec![Arc::new(BuiltinTool::new(
            ToolDesc::new(
                "temperature",
                "Get current temperature",
                ToolDescArg::new_object().with_properties(
                    [
                        (
                            "location",
                            ToolDescArg::new_string().with_desc("The city name"),
                        ),
                        (
                            "unit",
                            ToolDescArg::new_string().with_enum(["Celcius", "Fernheit"]),
                        ),
                    ],
                    ["location", "unit"],
                ),
                Some(
                    ToolDescArg::new_number()
                        .with_desc("Null if the given city name is unavailable."),
                ),
            ),
            Arc::new(|args| {
                if args
                    .as_object()
                    .unwrap()
                    .get("unit")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    == "Celcius"
                {
                    Part::Text("40".to_owned())
                } else {
                    Part::Text("104".to_owned())
                }
            }),
        )) as Arc<dyn Tool>];
        let mut agent = Agent::new(model, tools);

        let mut agg = MessageAggregator::new();
        let mut strm = Box::pin(agent.run("How much hot currently in Dubai?"));
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_mcp_stdio_tool_call() -> anyhow::Result<()> {
        use futures::StreamExt;

        use super::*;
        use crate::{model::LocalLanguageModel, tool::MCPTransport};

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!("{} / {}", progress.current_task, progress.total_task);
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = model.unwrap();

        let mut agent = Agent::new(model, vec![]);
        agent
            .add_mcp_tools(
                "time",
                MCPTransport::Stdio {
                    command: "uvx".into(),
                    args: vec!["mcp-server-time".into()],
                },
                vec![],
            )
            .await?;

        let agent_tools = agent.get_tools();
        assert_eq!(agent_tools.len(), 2);
        assert_eq!(
            agent_tools[0].get_description().name,
            "time--get_current_time"
        );
        assert_eq!(agent_tools[1].get_description().name, "time--convert_time");

        let mut agg = MessageAggregator::new();
        let mut strm = Box::pin(agent.run("What time is it now in America/New_York timezone?"));
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }

        Ok(())
    }
}
