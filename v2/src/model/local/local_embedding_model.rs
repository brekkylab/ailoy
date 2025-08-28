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
    utils::{BoxFuture, Mutex, Normalize},
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

#[cfg(any(target_family = "unix", target_family = "windows"))]
#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn dot(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        if a.len() != b.len() {
            panic!("Cannot dot two vectors of different lengths");
        }

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn infer_embedding() {
        let cache = crate::cache::Cache::new();
        let key = "BAAI/bge-m3";

        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(key));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = Arc::new(model.unwrap());

        let embedding = model.run("What is BGE M3?".to_owned()).await.unwrap();
        assert_eq!(embedding.len(), 1024);
    }

    #[tokio::test]
    async fn test_similariry() {
        use futures::StreamExt;

        use super::*;

        let cache = crate::cache::Cache::new();
        let key = "BAAI/bge-m3";

        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(key));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = Arc::new(model.unwrap());

        let query_embedding1 = model
            .clone()
            .run("What is BGE M3?".to_owned())
            .await
            .unwrap()
            .normalized();
        let query_embedding2 = model
            .clone()
            .run("Defination of BM25".to_owned())
            .await
            .unwrap()
            .normalized();
        let answer_embedding1 = model.clone().run("BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.".to_owned()).await.unwrap().normalized();
        let answer_embedding2 = model.clone().run("BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document".to_owned()).await.unwrap().normalized();
        assert!(
            dot(&query_embedding1, &answer_embedding1) > dot(&query_embedding1, &answer_embedding2)
        );
        assert!(
            dot(&query_embedding2, &answer_embedding1) < dot(&query_embedding2, &answer_embedding2)
        );
    }
}
