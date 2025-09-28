use std::{any::Any, collections::BTreeMap, sync::Arc};

use ailoy_macros::multi_platform_async_trait;
use anyhow::Result;
use futures::lock::Mutex;

use crate::{
    cache::{Cache, CacheClaim, CacheContents, CacheEntry, TryFromCache},
    dyn_maybe_send,
    knowledge_base::Embedding,
    model::{
        EmbeddingModelInference,
        local::{EmbeddingModelInferencer, Tokenizer},
    },
    utils::BoxFuture,
};

#[derive(Debug, Clone)]
pub(crate) struct LocalEmbeddingModel {
    tokenizer: Tokenizer,

    // The inferencer performs mutable operations such as KV cache updates, so the mutex is applied.
    inferencer: Arc<Mutex<EmbeddingModelInferencer>>,
}

#[multi_platform_async_trait]
impl EmbeddingModelInference for LocalEmbeddingModel {
    async fn infer(self: &mut Self, text: String) -> Result<Embedding> {
        let input_tokens = self.tokenizer.encode(&text, true).unwrap();
        let mut inferencer = self.inferencer.lock().await;

        #[cfg(target_family = "wasm")]
        let embedding = Ok(inferencer.infer(&input_tokens).await);
        #[cfg(not(target_family = "wasm"))]
        let embedding = inferencer.infer(&input_tokens).to_vec_f32();

        embedding
    }
}

impl TryFromCache for LocalEmbeddingModel {
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, Result<CacheClaim, String>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut tokenizer = Tokenizer::claim_files(cache.clone(), &key).await?;
            let mut inferncer = EmbeddingModelInferencer::claim_files(cache.clone(), &key).await?;
            let ctx: Box<dyn_maybe_send!(Any)> = Box::new([
                tokenizer
                    .entries
                    .iter()
                    .map(|v| v.clone())
                    .collect::<Vec<_>>(),
                inferncer
                    .entries
                    .iter()
                    .map(|v| v.clone())
                    .collect::<Vec<_>>(),
            ]);
            let mut rv = Vec::new();
            rv.append(&mut tokenizer.entries);
            rv.append(&mut inferncer.entries);
            Ok(CacheClaim::with_ctx(rv, ctx))
        })
    }

    fn try_from_contents(mut contents: CacheContents) -> BoxFuture<'static, Result<Self, String>>
    where
        Self: Sized,
    {
        Box::pin(async move {
            let (tok_entries, inf_entries) = match contents.ctx.take() {
                Some(ctx_any) => match ctx_any.downcast::<[Vec<CacheEntry>; 2]>() {
                    Ok(boxed) => {
                        let [a, b] = *boxed;
                        (a, b)
                    }
                    Err(_) => return Err("contents.ctx is not [Vec<CacheEntry>; 3]".into()),
                },
                None => return Err("contents.ctx is None".into()),
            };

            let tokenizer = {
                let mut files = BTreeMap::new();
                for k in tok_entries {
                    let v = contents.entries.remove(&k).unwrap();
                    files.insert(k, v);
                }
                let contents = CacheContents {
                    root: contents.root.clone(),
                    entries: files,
                    ctx: None,
                };
                Tokenizer::try_from_contents(contents).await?
            };
            let inferencer = {
                let mut files = BTreeMap::new();
                for k in inf_entries {
                    let v = contents.entries.remove(&k).unwrap();
                    files.insert(k, v);
                }
                let contents = CacheContents {
                    root: contents.root.clone(),
                    entries: files,
                    ctx: None,
                };
                EmbeddingModelInferencer::try_from_contents(contents).await?
            };

            Ok(LocalEmbeddingModel {
                tokenizer,
                inferencer: Arc::new(Mutex::new(inferencer)),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::StreamExt;

    use super::*;
    use crate::utils::{Normalize, log};

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            panic!("Cannot dot two vectors of different lengths");
        }

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[multi_platform_test]
    async fn infer_embedding() {
        let cache = crate::cache::Cache::new();
        let key = "BAAI/bge-m3";

        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(key));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            log::info(format!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            ));
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let mut model = model.unwrap();

        let embedding = model.infer("What is BGE M3?".to_owned()).await.unwrap();
        assert_eq!(embedding.len(), 1024);
        log::debug(format!("{:?}", embedding.normalized()));
    }

    #[multi_platform_test]
    async fn check_similarity() {
        use futures::StreamExt;

        use super::*;

        let cache = crate::cache::Cache::new();
        let key = "BAAI/bge-m3";

        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(key));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            log::info(format!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            ));
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let mut model = model.unwrap();

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
        assert!(
            dot(&query_embedding1, &answer_embedding1) > dot(&query_embedding1, &answer_embedding2)
        );
        assert!(
            dot(&query_embedding2, &answer_embedding1) < dot(&query_embedding2, &answer_embedding2)
        );
        log::debug(format!("{:?}", dot(&query_embedding1, &answer_embedding1)));
        log::debug(format!("{:?}", dot(&query_embedding1, &answer_embedding2)));
        log::debug(format!("{:?}", dot(&query_embedding2, &answer_embedding1)));
        log::debug(format!("{:?}", dot(&query_embedding2, &answer_embedding2)));
    }
}
