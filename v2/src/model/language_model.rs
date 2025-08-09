use std::pin::Pin;

use futures::Stream;

use crate::{
    model::{APILanguageModel, LocalLanguageModel},
    value::{Message, MessageDelta, ToolDescription},
};

pub trait LanguageModel: Clone {
    fn run(
        self,
        tools: Vec<ToolDescription>,
        msg: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = Result<MessageDelta, String>>>>;
}

#[derive(Clone, Debug)]
pub enum AnyLanguageModel {
    API(APILanguageModel),
    Local(LocalLanguageModel),
}

impl LanguageModel for AnyLanguageModel {
    fn run(
        self,
        tools: Vec<ToolDescription>,
        msgs: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = Result<MessageDelta, String>>>> {
        match self {
            AnyLanguageModel::API(m) => m.run(tools, msgs),
            AnyLanguageModel::Local(m) => m.run(tools, msgs),
        }
    }
}
