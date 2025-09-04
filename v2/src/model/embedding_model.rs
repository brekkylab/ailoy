use anyhow::Result;

use crate::{
    knowledge_base::Embedding,
    utils::{MaybeSend, MaybeSync},
};
use ailoy_macros::multi_platform_async_trait;

#[multi_platform_async_trait]
pub trait EmbeddingModel: MaybeSend + MaybeSync + 'static {
    async fn run(self: &mut Self, text: String) -> Result<Embedding>;
}
