use ailoy_macros::multi_platform_async_trait;
use anyhow::Result;
use futures::{Stream, StreamExt as _};

use crate::{
    cache::{Cache, CacheProgress},
    knowledge_base::Embedding,
    model::local::LocalEmbeddingModel,
    utils::{MaybeSend, MaybeSync},
};

#[multi_platform_async_trait]
pub trait EmbeddingModelInference: MaybeSend + MaybeSync {
    async fn infer(self: &mut Self, text: String) -> Result<Embedding>;
}

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    inner: LocalEmbeddingModel,
}

impl EmbeddingModel {
    pub async fn try_new(
        model_name: impl Into<String>,
    ) -> impl Stream<Item = Result<CacheProgress<Self>, String>> + 'static {
        let model_name = model_name.into();
        let mut strm = Box::pin(Cache::new().try_create::<LocalEmbeddingModel>(model_name));
        async_stream::try_stream! {
            while let Some(result) = strm.next().await {
                let result = result?;
                yield CacheProgress {
                    comment: result.comment,
                    current_task: result.current_task,
                    total_task: result.current_task,
                    result: result.result.map(|v| EmbeddingModel{inner: v}),
                };
            }
        }
    }
}

#[multi_platform_async_trait]
impl EmbeddingModelInference for EmbeddingModel {
    async fn infer(self: &mut Self, text: String) -> Result<Embedding> {
        self.inner.infer(text).await
    }
}
