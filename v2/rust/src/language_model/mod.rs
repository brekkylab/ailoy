mod api;
mod local;

use std::pin::Pin;

use futures::Stream;

use crate::{Message, MessageDelta};

pub use api::APILanguageModel;
pub use local::LocalLanguageModel;

pub trait LanguageModel: Clone {
    fn run(self, msg: Vec<Message>) -> Pin<Box<dyn Stream<Item = Result<MessageDelta, String>>>>;
}

#[derive(Clone, Debug)]
pub enum AnyLanguageModel {
    Api(APILanguageModel),
    Local(LocalLanguageModel),
}

impl LanguageModel for AnyLanguageModel {
    fn run(self, msgs: Vec<Message>) -> Pin<Box<dyn Stream<Item = Result<MessageDelta, String>>>> {
        match self {
            AnyLanguageModel::Api(m) => m.run(msgs),
            AnyLanguageModel::Local(m) => m.run(msgs),
        }
    }
}
