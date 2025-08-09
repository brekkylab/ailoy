use crate::{
    model::LanguageModel,
    value::{Message, MessageDelta, ToolDescription},
};

#[derive(Debug, Clone)]
pub struct APILanguageModel {}

impl LanguageModel for APILanguageModel {
    fn run(
        self,
        _tools: Vec<ToolDescription>,
        _msg: Vec<Message>,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<MessageDelta, String>>>> {
        todo!()
    }
}
