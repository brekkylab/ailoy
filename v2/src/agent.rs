use std::sync::Arc;

use futures::{Stream, StreamExt};
use tokio::sync::Mutex;

use crate::{
    model::LanguageModel,
    tool::Tool,
    value::{Message, MessageAggregator, MessageOutput, Part, Role},
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
    ) -> impl Stream<Item = Result<MessageOutput, String>> {
        let lm = self.lm.clone();
        let tools = self.tools.clone();
        let msgs = self.messages.clone();
        let user_message =
            Message::new(Role::User).with_contents(vec![Part::Text(user_message.into())]);
        async_stream::try_stream! {
            msgs.lock().await.push(user_message);
            loop {
                let td = self.tools.iter().map(|v| v.get_description()).collect::<Vec<_>>();
                let mut strm = lm.clone().run(msgs.lock().await.clone(), td);
                let mut aggregator = MessageAggregator::new();
                let mut assistant_msg = Message::new(Role::Assistant);
                while let Some(delta) = strm.next().await {
                    let delta = delta?;
                    yield delta.clone();
                    if let Some(msg) = aggregator.update(delta) {
                        assistant_msg = msg;
                    }
                }
                self.messages.lock().await.push(assistant_msg.clone());
                if !assistant_msg.tool_calls.is_empty() {
                    for part in assistant_msg.tool_calls {
                        todo!()
                        // let tc = ToolCall::try_from_string(part.get_function_owned().unwrap()).unwrap();
                        // let tool = tools.iter().find(|v| v.get_description().get_name() == tc.get_name()).unwrap().clone();
                        // let resp = tool.run(tc).await?;
                        // let delta = MessageDelta::new().with_role(Role::Tool).with_content(vec![resp.clone()]);
                        // yield MessageOutput::new().with_delta(delta);
                        // let tool_msg = Message::new(Role::Tool).with_content(vec![resp]);
                        // self.messages.lock().await.push(tool_msg);
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
            Arc::new(|tc| {
                if tc
                    .arguments
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
}
