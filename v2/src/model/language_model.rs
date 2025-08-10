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

impl AnyLanguageModel {
    pub fn from_api_model(m: APILanguageModel) -> Self {
        AnyLanguageModel::API(m)
    }

    pub fn from_local_model(m: LocalLanguageModel) -> Self {
        AnyLanguageModel::Local(m)
    }
}

impl From<APILanguageModel> for AnyLanguageModel {
    fn from(m: APILanguageModel) -> Self {
        AnyLanguageModel::from_api_model(m)
    }
}

impl From<LocalLanguageModel> for AnyLanguageModel {
    fn from(m: LocalLanguageModel) -> Self {
        AnyLanguageModel::from_local_model(m)
    }
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
