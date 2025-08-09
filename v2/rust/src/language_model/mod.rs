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
