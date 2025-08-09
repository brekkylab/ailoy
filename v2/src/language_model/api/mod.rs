use crate::{
    language_model::LanguageModel,
    message::{Message, MessageDelta, ToolDescription},
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
