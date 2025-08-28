use anyhow::Result;
use std::sync::Arc;

use crate::{
    knowledge_base::Embedding,
    utils::{MaybeSend, MaybeSync},
};
use ailoy_macros::multi_platform_async_trait;

#[multi_platform_async_trait]
pub trait EmbeddingModel: MaybeSend + MaybeSync + 'static {
    // Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    /// See [`LanguageModel`] trait document for the details.
    async fn run(self: Arc<Self>, text: String) -> Result<Embedding>;
}
