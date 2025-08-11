use std::sync::Arc;

use futures::stream::BoxStream;

use crate::{
    model::LanguageModel,
    value::{Message, MessageDelta, ToolDescription},
};

#[derive(Debug, Clone)]
pub struct APILanguageModel {}

impl LanguageModel for APILanguageModel {
    fn run(
        self: Arc<Self>,
        _tools: Vec<ToolDescription>,
        _msg: Vec<Message>,
    ) -> BoxStream<'static, Result<MessageDelta, String>> {
        todo!()
    }
}
