use std::collections::HashMap;
use std::sync::Arc;

use anyhow::anyhow;
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
    mcp_clients: HashMap<String, Arc<MCPClient>>,
}

impl Agent {
    pub fn new(lm: impl LanguageModel, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        Self {
            lm: Arc::new(lm),
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
            mcp_clients: HashMap::new(),
        }
    }

    pub fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    pub async fn add_mcp_client(
        &mut self,
        name: &str,
        client: MCPClient,
        tools_to_add: Vec<String>,
    ) -> anyhow::Result<()> {
        if self.mcp_clients.contains_key(name) {
            return Err(anyhow!(
                "MCP client with \"{}\" is already registered",
                name
            ));
        }

        // Add client to self.mcp_clients
        let client = Arc::new(client);
        self.mcp_clients.insert(name.into(), client.clone());

        // Add tools in the client
        let tools = client.list_tools().await?;
        for mut tool in tools.into_iter() {
            // If tools_to_add is not empty, check if this tool is in the whitelist
            if tools_to_add.len() > 0 && !tools_to_add.contains(&tool.desc.name) {
                continue;
            }
            tool.desc.name = format!("{}--{}", name, tool.desc.name);
            self.tools.push(Arc::new(tool) as Arc<dyn Tool>);
        }

        Ok(())
    }

    pub async fn remove_mcp_client(&mut self, name: &str) -> anyhow::Result<()> {
        if !self.mcp_clients.contains_key(name) {
            return Err(anyhow!(
                "MCP client with \"{}\" is not registered in this agent",
                name
            ));
        }

        // Remove tools in the client
        let client = self.mcp_clients.get(name).unwrap();
        let mcp_tool_names = client
            .list_tools()
            .await?
            .iter()
            .map(|t| format!("{}--{}", name, t.desc.name))
            .collect::<Vec<String>>();
        self.tools.retain(|t| {
            let desc = t.get_description();
            !mcp_tool_names.contains(&desc.name)
        });

        // Remove MCP client
        self.mcp_clients.remove(name);

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
        agent.add_mcp_client("time", client, vec![]).await?;

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
