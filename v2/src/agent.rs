use std::collections::HashMap;

use futures::StreamExt;

use crate::{
    model::{AnyLanguageModel, LanguageModel as _},
    tool::{AnyTool, Tool},
    value::{Message, MessageAggregator, Role, ToolCall},
};

pub struct Agent {
    lm: AnyLanguageModel,
    tools: HashMap<String, AnyTool>,
    messages: Vec<Message>,
}

impl Agent {
    pub async fn run(&mut self, user_message: Message) -> Result<(), String> {
        self.messages.push(user_message);

        loop {
            let tools = self
                .tools
                .iter()
                .map(|(_, v)| v.get_description())
                .collect::<Vec<_>>();
            let mut strm = self.lm.clone().run(tools, self.messages.clone());
            let mut aggregator = MessageAggregator::new(Role::Assistant);
            while let Some(delta) = strm.next().await {
                aggregator.update(delta?);
            }
            let assistant_message = aggregator.finalize();
            if !assistant_message.tool_calls().is_empty() {
                for part in assistant_message.tool_calls() {
                    let _ = ToolCall::try_from_string(part.get_json_owned().unwrap()).unwrap();
                    todo!()
                }
            } else {
                break;
            }
        }
        Ok(())
    }
}
