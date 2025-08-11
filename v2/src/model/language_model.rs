use std::sync::Arc;

use futures::stream::BoxStream;

use crate::value::{Message, MessageDelta, ToolDescription};

pub trait LanguageModel: Send + Sync + 'static {
    fn run(
        self: Arc<Self>,
        tools: Vec<ToolDescription>,
        msg: Vec<Message>,
    ) -> BoxStream<'static, Result<MessageDelta, String>>;
}
