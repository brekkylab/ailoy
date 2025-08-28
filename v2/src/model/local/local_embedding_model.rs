use anyhow::Result;
use futures::stream::Stream;
use std::sync::Arc;

use crate::{
    cache::{Cache, CacheContents, CacheEntry, CacheProgress, TryFromCache},
    knowledge_base::Embedding,
    model::{
        EmbeddingModel,
        local::{EmbeddingModelInferencer, Tokenizer},
    },
    utils::{BoxFuture, Mutex},
};
use ailoy_macros::multi_platform_async_trait;

#[derive(Debug)]
pub struct LocalEmbeddingModel {
    tokenizer: Tokenizer,

    // The inferencer performs mutable operations such as KV cache updates, so the mutex is applied.
    inferencer: Mutex<EmbeddingModelInferencer>,
}

impl LocalEmbeddingModel {
    pub async fn try_new(
        model_name: impl Into<String>,
    ) -> impl Stream<Item = Result<CacheProgress<Self>, String>> + 'static {
        let model_name = model_name.into();
        Cache::new().try_create::<LocalEmbeddingModel>(model_name)
    }
}

#[multi_platform_async_trait]
impl EmbeddingModel for LocalEmbeddingModel {
    async fn run(self: Arc<Self>, text: String) -> Result<Embedding> {
        let input_tokens = self.tokenizer.encode(&text, true).unwrap();
        let tensor = self.inferencer.lock().infer(&input_tokens);
        return tensor.to_vec_f32();
    }
}

impl TryFromCache for LocalEmbeddingModel {
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut rv = Vec::new();
            rv.append(&mut Tokenizer::claim_files(cache.clone(), &key).await?);
            rv.append(&mut EmbeddingModelInferencer::claim_files(cache.clone(), &key).await?);
            Ok(rv)
        })
    }

    fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String>
    where
        Self: Sized,
    {
        let tokenizer = Tokenizer::try_from_contents(contents)?;
        let inferencer = EmbeddingModelInferencer::try_from_contents(contents)?;
        Ok(LocalEmbeddingModel {
            tokenizer,
            inferencer: Mutex::new(inferencer),
        })
    }
}
