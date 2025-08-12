use std::sync::Arc;

use futures::{Stream, StreamExt};
use tokio::sync::Mutex;

use crate::{
    model::LanguageModel,
    tool::Tool,
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
                while let Some(delta) = strm.next().await {
                    let delta = delta?;
                    yield delta.clone();
                    aggregator.update(delta);
                }
                let assistant_msg = aggregator.finalize().unwrap();
                self.messages.lock().await.push(assistant_msg.clone());
                if !assistant_msg.tool_calls.is_empty() {
                    for part in assistant_msg.tool_calls {
                        let tc = ToolCall::try_from_string(part.get_function_owned().unwrap()).unwrap();
                        let tool = tools.iter().find(|v| v.get_description().get_name() == tc.get_name()).unwrap().clone();
                        let resp = tool.run(tc).await?;
                        yield MessageDelta::Content(Role::Tool, resp.clone());
                        let tool_msg = Message::with_content(Role::Tool, resp);
                        self.messages.lock().await.push(tool_msg);
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
            let progress = progress.unwrap();
            println!("{} / {}", progress.current_task(), progress.total_task());
            if progress.current_task() == progress.total_task() {
                model = progress.take();
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
            let progress = progress.unwrap();
            println!("{} / {}", progress.current_task(), progress.total_task());
            if progress.current_task() == progress.total_task() {
                model = progress.take();
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
}
