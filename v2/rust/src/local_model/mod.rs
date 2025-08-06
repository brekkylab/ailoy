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
        let mut files: Vec<PathBuf> = Vec::new();
        let key = "Qwen/Qwen3-0.6B";

        let v = ChatTemplate::claim_files(cache.clone(), key.to_owned())
            .await
            .unwrap();
        files.extend(v);

        let v = Tokenizer::claim_files(cache.clone(), key.to_owned())
            .await
            .unwrap();
        files.extend(v);

        let v = TVMModel::claim_files(cache.clone(), key.to_owned())
            .await
            .unwrap();
        files.extend(v);

        println!("{:?}", files);
    }
}
