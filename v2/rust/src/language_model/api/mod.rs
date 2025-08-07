use crate::language_model::LanguageModel;

#[derive(Debug)]
pub struct APILanguageModel {}

impl LanguageModel for APILanguageModel {
    fn run(
        &self,
        _msg: &Vec<crate::Message>,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<crate::MessageDelta, String>>>> {
        todo!()
    }
}
