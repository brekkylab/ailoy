mod chat_template;
mod inferencer;
mod tokenizer;

pub use chat_template::*;
pub use inferencer::*;
pub use tokenizer::*;

use crate::language_model::LanguageModel;

#[derive(Debug)]
pub struct LocalLanguageModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: Tokenizer,
    inferencer: Inferencer,
}

impl<'a> LanguageModel for LocalLanguageModel<'a> {
    fn run(
        &self,
        _msg: &Vec<crate::Message>,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<crate::MessageDelta, String>>>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn test1() {
        let cache = crate::cache::Cache::new();
        let mut files: Vec<PathBuf> = Vec::new();
        let key = "Qwen/Qwen3-0.6B";
        let model = cache.try_create::<Inferencer>(key).await.unwrap();
    }
}
