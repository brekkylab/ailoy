use std::sync::Arc;

use futures::{Stream, StreamExt};
use tokio::sync::Mutex;

use crate::{
    model::LanguageModel,
    tool::{MCPClient, Tool},
    value::{Message, MessageAggregator, MessageDelta, Part, Role, ToolCall},
};

pub struct Agent {
    lm: Arc<dyn LanguageModel>,
    tools: Vec<Arc<dyn Tool>>,
    messages: Arc<Mutex<Vec<Message>>>,
}

impl Agent {
    pub fn new(lm: impl LanguageModel, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        Self {
            lm: Arc::new(lm),
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    pub async fn add_mcp_tools(
        &mut self,
        client_name: &str,
        client: MCPClient,
        tools_to_add: Vec<String>,
    ) -> anyhow::Result<()> {
        let mut tools = client.list_tools().await?;

        // If tools_to_add is not empty, filter out the tools not in the whitelist.
        if !tools_to_add.is_empty() {
            tools.retain(|t| tools_to_add.contains(&t.get_description().name))
        }

        for mut tool in tools.into_iter() {
            // The name of MCP tool description is prefixed with the provided client name.
            let tool_desc_name = format!("{}--{}", client_name, tool.desc.name);

            // If the tool with same name already exists, skip adding the tool.
            if self
                .tools
                .iter()
                .find(|t| t.get_description().name == tool_desc_name)
                .is_some()
            {
                println!(
                    "MCP tool \"{}\" is already registered. Skip adding the tool.",
                    tool_desc_name
                );
                continue;
            }

            tool.desc.name = tool_desc_name;
            self.tools.push(Arc::new(tool) as Arc<dyn Tool>);
        }

        Ok(())
    }

    pub async fn remove_mcp_tools(&mut self, client_name: &str) -> anyhow::Result<()> {
        // Remove MCP tools
        self.tools.retain(|t| {
            let tool_desc_name = t.get_description().name;
            // Remove the MCP tool if its description name is prefixed with the provided client name.
            !tool_desc_name.starts_with(format!("{}--", client_name).as_str())
        });

        Ok(())
    }

    pub fn run(
        &mut self,
        user_message: impl Into<String>,
    ) -> impl Stream<Item = Result<MessageDelta, String>> {
        let lm = self.lm.clone();
        let tools = self.tools.clone();
        let msgs = self.messages.clone();
        let user_message = Message::with_content(Role::User, Part::Text(user_message.into()));
        async_stream::try_stream! {
            msgs.lock().await.push(user_message);
            loop {
                let td = self.tools.iter().map(|v| v.get_description()).collect::<Vec<_>>();
                let mut strm = lm.clone().run(td, msgs.lock().await.clone());
                let mut aggregator = MessageAggregator::new();
                let mut assistant_msg: Option<Message> = None;
                while let Some(delta) = strm.next().await {
                    let delta = delta?;
                    yield delta.clone();
                    if let Some(msg) = aggregator.update(delta) {
                        assistant_msg = Some(msg);
                    }
                }
                if let Some(msg) = aggregator.finalize() {
                        assistant_msg = Some(msg);
                    }
                let assistant_msg = assistant_msg.unwrap();
                self.messages.lock().await.push(assistant_msg.clone());
                if !assistant_msg.tool_calls.is_empty() {
                    for part in assistant_msg.tool_calls {
                        let tc = ToolCall::try_from_string(part.get_function_owned().unwrap()).unwrap();
                        let tool = tools.iter().find(|v| v.get_description().get_name() == tc.get_name()).unwrap().clone();
                        let parts = tool.run(tc).await?;
                        for part in parts.into_iter() {
                            yield MessageDelta::Content(Role::Tool, part.clone());
                            let tool_msg = Message::with_content(Role::Tool, part);
                            self.messages.lock().await.push(tool_msg);
                        }
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
        if let Some(msg) = agg.finalize() {
            println!("{:?}", msg);
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
            value::{ToolDescription, ToolDescriptionArgument},
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
            ToolDescription::new(
                "temperature",
                "Get current temperature",
                ToolDescriptionArgument::new_object().with_properties(
                    [
                        (
                            "location",
                            ToolDescriptionArgument::new_string().with_desc("The city name"),
                        ),
                        (
                            "unit",
                            ToolDescriptionArgument::new_string()
                                .with_enum(["Celcius", "Fernheit"]),
                        ),
                    ],
                    ["location", "unit"],
                ),
                Some(
                    ToolDescriptionArgument::new_number()
                        .with_desc("Null if the given city name is unavailable."),
                ),
            ),
            Arc::new(|tc| {
                if tc
                    .get_argument()
                    .as_object()
                    .unwrap()
                    .get("unit")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    == "Celcius"
                {
                    Part::new_text("40")
                } else {
                    Part::new_text("104")
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
        if let Some(msg) = agg.finalize() {
            println!("{:?}", msg);
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_mcp_stdio_tool_call() -> anyhow::Result<()> {
        use futures::StreamExt;

        use super::*;
        use crate::model::LocalLanguageModel;
        use crate::tool::MCPClient;
        use rmcp::transport::ConfigureCommandExt;

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

        let client = MCPClient::from_stdio(tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        }))
        .await?;

        let mut agent = Agent::new(model, vec![]);
        agent.add_mcp_tools("time", client, vec![]).await?;

        let agent_tools = agent.get_tools();
        assert_eq!(agent_tools.len(), 2);
        assert_eq!(
            agent_tools[0].get_description().name,
            "time--get_current_time"
        );
        assert_eq!(agent_tools[1].get_description().name, "time--convert_time");

        let mut agg = MessageAggregator::new();
        let mut strm = Box::pin(agent.run("What time is it now in Asia/Seoul?"));
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
        if let Some(msg) = agg.finalize() {
            println!("{:?}", msg);
        }

        Ok(())
    }
}
