use std::{collections::HashMap, sync::Arc};

use ailoy_macros::multi_platform_async_trait;
use async_stream::try_stream;
use futures::{StreamExt as _, lock::Mutex};

use crate::{
    boxed,
    cache::{Cache, CacheClaim, CacheContents, CacheProgress, TryFromCache},
    model::{
        EmbeddingModelInference,
        local::{EmbeddingModelInferencer, Tokenizer},
    },
    utils::{BoxFuture, BoxStream, Normalize},
    value::{Embedding, Value},
};

#[derive(Debug, Clone)]
pub(crate) struct LocalEmbeddingModel {
    tokenizer: Tokenizer,

    // The inferencer performs mutable operations such as KV cache updates, so the mutex is applied.
    inferencer: Arc<Mutex<EmbeddingModelInferencer>>,

    do_normalize: bool,
}

#[derive(Clone, Debug, Default)]
pub struct LocalEmbeddingModelConfig {
    pub device_id: Option<i32>,
    pub validate_checksum: Option<bool>,
}

#[multi_platform_async_trait]
impl EmbeddingModelInference for LocalEmbeddingModel {
    async fn infer(&self, text: String) -> anyhow::Result<Embedding> {
        let input_tokens = self.tokenizer.encode(&text, true).unwrap();
        let mut inferencer = self.inferencer.lock().await;

        #[cfg(target_family = "wasm")]
        let mut embedding = inferencer.infer(&input_tokens).await;
        #[cfg(not(target_family = "wasm"))]
        let mut embedding = inferencer.infer(&input_tokens).to_vec_f32()?;

        if self.do_normalize {
            embedding = embedding.normalized();
        }

        Ok(embedding.into())
    }
}

impl TryFromCache for LocalEmbeddingModel {
    fn claim_files<'a>(
        cache: Cache,
        key: impl AsRef<str>,
        ctx: &'a mut std::collections::HashMap<String, crate::value::Value>,
    ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut tokenizer_claim = Tokenizer::claim_files(cache.clone(), &key, ctx).await?;
            let mut tokenizer_entries = Vec::new();
            for entry in tokenizer_claim.entries.iter() {
                tokenizer_entries.push(crate::to_value!({
                    dirname: entry.dirname().to_owned(),
                    filename: entry.filename().to_owned()
                }));
            }
            ctx.insert("tokenizer_entries".to_owned(), tokenizer_entries.into());

            let mut inferencer_claim =
                EmbeddingModelInferencer::claim_files(cache.clone(), &key, ctx).await?;
            let mut inferencer_entries = Vec::new();
            for entry in inferencer_claim.entries.iter() {
                inferencer_entries.push(crate::to_value!({
                    dirname: entry.dirname().to_owned(),
                    filename: entry.filename().to_owned()
                }));
            }
            ctx.insert("inferencer_entries".to_owned(), inferencer_entries.into());

            let mut rv = Vec::new();
            rv.append(&mut tokenizer_claim.entries);
            rv.append(&mut inferencer_claim.entries);
            Ok(CacheClaim::new(rv))
        })
    }

    fn try_from_contents<'a>(
        contents: &'a mut CacheContents,
        ctx: &std::collections::HashMap<String, crate::value::Value>,
    ) -> BoxFuture<'a, anyhow::Result<Self>>
    where
        Self: Sized,
    {
        let ctx = ctx.to_owned();
        Box::pin(async move {
            let tokenizer = Tokenizer::try_from_contents(contents, &ctx).await?;
            let inferencer = EmbeddingModelInferencer::try_from_contents(contents, &ctx).await?;
            Ok(LocalEmbeddingModel {
                tokenizer,
                inferencer: Arc::new(Mutex::new(inferencer)),
                do_normalize: true,
            })
        })
    }
}

impl LocalEmbeddingModel {
    pub async fn try_new(
        model: impl Into<String>,
        config: Option<LocalEmbeddingModelConfig>,
    ) -> anyhow::Result<Self> {
        let config = config.unwrap_or_default();
        let cache = Cache::new();
        let mut ctx = HashMap::new();
        if let Some(device_id) = config.device_id {
            ctx.insert("device_id".to_owned(), Value::integer(device_id.into()));
        };
        let mut strm =
            Box::pin(cache.try_create::<Self>(model, Some(ctx), config.validate_checksum));
        while let Some(v) = strm.next().await {
            if let Some(result) = v?.result {
                return Ok(result);
            }
        }
        unreachable!()
    }

    pub fn try_new_stream<'a>(
        model: impl Into<String>,
        config: Option<LocalEmbeddingModelConfig>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<Self>>> {
        let config = config.unwrap_or_default();
        let cache = Cache::new();
        let mut ctx = HashMap::new();
        if let Some(device_id) = config.device_id {
            ctx.insert("device_id".to_owned(), Value::integer(device_id.into()));
        };
        let strm = cache.try_create::<Self>(model, Some(ctx), config.validate_checksum);
        boxed!(strm)
    }

    pub fn download<'a>(
        model: impl Into<String>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<()>>> {
        let cache = Cache::new();
        let mut strm = cache.prepare_files::<Self>(model, Some(true));
        boxed!(try_stream! {
            while let Some(res) = strm.next().await {
                let (entry, current_task, total_task, _) = res?;
                yield CacheProgress::<()> {
                    comment: format!("{} downloaded", entry.filename()),
                    current_task,
                    total_task,
                    result: None,
                }
            }
        })
    }

    pub async fn remove(model: impl Into<String>) -> anyhow::Result<()> {
        let cache = Cache::new();
        let model = model.into();
        let claim = Self::claim_files(cache.clone(), &model, &mut HashMap::new())
            .await
            .expect(format!("Failed to get the entries for {}", model).as_str());

        for entry in claim.entries.iter() {
            cache.remove(entry).await.unwrap();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::StreamExt;

    use super::*;

    #[multi_platform_test]
    async fn infer_embedding() {
        let cache = Cache::new();
        let key = "BAAI/bge-m3";

        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(key, None, None));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            crate::info!(
                "{} ({} / {})",
                progress.comment,
                progress.current_task,
                progress.total_task
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = model.unwrap();

        let embedding = model.infer("What is BGE M3?".to_owned()).await.unwrap();
        assert_eq!(embedding.len(), 1024);
        crate::debug!("{:?}", embedding.normalized());
    }

    #[multi_platform_test]
    async fn check_similarity() {
        use futures::StreamExt;

        use super::*;

        let cache = Cache::new();
        let key = "BAAI/bge-m3";

        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(key, None, None));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            crate::info!(
                "{} ({} / {})",
                progress.comment,
                progress.current_task,
                progress.total_task,
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let model = model.unwrap();

        let query_embedding1 = model
            .infer("What is BGE M3?".to_owned())
            .await
            .unwrap()
            .normalized();
        let query_embedding2 = model
            .infer("Defination of BM25".to_owned())
            .await
            .unwrap()
            .normalized();
        let answer_embedding1 = model.infer("BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.".to_owned()).await.unwrap().normalized();
        let answer_embedding2 = model.infer("BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document".to_owned()).await.unwrap().normalized();
        assert!(&query_embedding1 * &answer_embedding1 > &query_embedding1 * &answer_embedding2);
        assert!(&query_embedding2 * &answer_embedding1 < &query_embedding2 * &answer_embedding2);
    }
}
