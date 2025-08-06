mod chat_template;
mod tokenizer;
mod tvm_model;

pub use chat_template::*;
pub use tokenizer::*;
pub use tvm_model::*;

#[derive(Debug)]
pub struct Inferencer {}

#[derive(Debug)]
pub struct LocalLangModel<'a> {
    pub chat_template: ChatTemplate<'a>,
    pub tokenizer: Tokenizer,
    pub inferencer: Inferencer,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::cache::TryFromCache;

    use super::*;

    #[tokio::test]
    async fn test1() {
        let cache = crate::cache::Cache::new();
        // let mut files: Vec<PathBuf> = Vec::new();
        let key = "Qwen/Qwen3-0.6B";

        // let ct = cache
        //     .try_create_from_cache::<ChatTemplate>(key)
        //     .await
        //     .unwrap();

        // let tok = cache.try_create_from_cache::<Tokenizer>(key).await.unwrap();

        let model = cache.try_create_from_cache::<TVMModel>(key).await.unwrap();
    }
}
